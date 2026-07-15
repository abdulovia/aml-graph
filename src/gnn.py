"""GraphSAGE edge classifier for AML with leakage-safe training.

Improvements over a naive edge-GNN, validated experimentally:

* **Feature standardisation** (fit on the training slice only) — the motif edge
  features and node-degree features span very different scales; without scaling
  the MLP head is unstable and the GNN badly underperforms.
* **LayerNorm** between SAGE layers for stable optimisation.
* **Early stopping** on a temporal validation slice (PR-AUC), so the model does
  not over-train.
* **Leakage-safe inductive inference**: message passing during training uses
  only the inner-train edges; validation edges are excluded from the graph when
  scoring validation; test edges are scored on a graph built from *all* train
  edges (strictly in their past). No future information reaches an edge's
  endpoints.
* Reverse edges are added (Multi-GNN "reverse message passing").

Experiments also tried an edge-aware GINEConv and a LightGBM⊕GNN ensemble; both
were recorded but did not beat the interpretable LightGBM baseline on this
subsample, so the simpler, faster SAGE variant is the default.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch import Tensor, nn
from torch_geometric.nn import SAGEConv

from src.config import SEED


class SAGEEdgeClassifier(nn.Module):
    """Two-layer GraphSAGE encoder (LayerNorm) + MLP edge-scoring head."""

    def __init__(self, n_node_feats: int, n_edge_feats: int, hidden: int = 96) -> None:
        """Initialise encoder and head.

        Args:
            n_node_feats: Node feature dimension.
            n_edge_feats: Per-edge motif feature dimension.
            hidden: Hidden width.

        """
        super().__init__()
        self.conv1 = SAGEConv(n_node_feats, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + n_edge_feats, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Compute node embeddings via message passing."""
        h = F.relu(self.norm1(self.conv1(x, edge_index)))
        h = F.relu(self.norm2(self.conv2(h, edge_index)))
        return h

    def forward(self, x: Tensor, edge_index: Tensor, pairs: Tensor, edge_attr: Tensor) -> Tensor:
        """Return per-edge logits for the endpoint pairs in ``pairs``."""
        h = self.encode(x, edge_index)
        z = torch.cat([h[pairs[0]], h[pairs[1]], edge_attr], dim=1)
        return self.head(z).squeeze(-1)


def _node_features(su: np.ndarray, dv: np.ndarray, mp_idx: np.ndarray, n_nodes: int) -> np.ndarray:
    """Degree/volume node features computed from a message-passing edge set."""
    in_deg = np.zeros(n_nodes, dtype=np.float32)
    out_deg = np.zeros(n_nodes, dtype=np.float32)
    for i in mp_idx:
        out_deg[su[i]] += 1
        in_deg[dv[i]] += 1
    return np.stack([np.log1p(in_deg), np.log1p(out_deg), (in_deg > 0).astype(np.float32)], axis=1)


def _edge_index(su: np.ndarray, dv: np.ndarray, mp_idx: np.ndarray) -> Tensor:
    """Directed edge_index with added reverse edges (Multi-GNN style)."""
    fwd = np.stack([su[mp_idx], dv[mp_idx]])
    rev = np.stack([dv[mp_idx], su[mp_idx]])
    return torch.tensor(np.concatenate([fwd, rev], axis=1), dtype=torch.long)


def train_gnn(
    src: list[str],
    dst: list[str],
    edge_attr: np.ndarray,
    y: np.ndarray,
    tr_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    epochs: int = 200,
    lr: float = 3e-3,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Train the edge classifier and score validation + test edges.

    Args:
        src: Source node id per edge.
        dst: Destination node id per edge.
        edge_attr: Per-edge motif feature matrix (unscaled).
        y: Binary illicit labels.
        tr_idx: Inner-train indices (message passing + loss).
        val_idx: Validation indices (early stopping + threshold).
        test_idx: Test indices (final scoring).
        epochs: Max epochs (early stopping usually stops earlier).
        lr: Learning rate.
        device: Torch device.

    Returns:
        ``(val_scores, test_scores)`` sigmoid probabilities aligned with
        ``val_idx`` / ``test_idx``.

    """
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    dev = torch.device(device)

    nodes = sorted(set(src) | set(dst))
    idx_of = {n: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)
    su = np.array([idx_of[s] for s in src], dtype=np.int64)
    dv = np.array([idx_of[d] for d in dst], dtype=np.int64)
    train_all = np.concatenate([tr_idx, val_idx])

    # standardise edge features on inner-train only (no leakage)
    mu, sd = edge_attr[tr_idx].mean(0), edge_attr[tr_idx].std(0) + 1e-6
    eattr = torch.tensor((edge_attr - mu) / sd, dtype=torch.float32).to(dev)

    # node features + scaler fit on the inner-train message-passing graph
    nf_tr = _node_features(su, dv, tr_idx, n_nodes)
    nmu, nsd = nf_tr.mean(0), nf_tr.std(0) + 1e-6
    x_tr = torch.tensor((nf_tr - nmu) / nsd, dtype=torch.float32).to(dev)
    nf_all = _node_features(su, dv, train_all, n_nodes)
    x_all = torch.tensor((nf_all - nmu) / nsd, dtype=torch.float32).to(dev)

    ei_tr = _edge_index(su, dv, tr_idx).to(dev)
    ei_all = _edge_index(su, dv, train_all).to(dev)
    pairs = torch.tensor(np.stack([su, dv]), dtype=torch.long).to(dev)
    yt = torch.tensor(y, dtype=torch.float32).to(dev)
    tr_t = torch.tensor(tr_idx).to(dev)
    val_t = torch.tensor(val_idx).to(dev)

    model = SAGEEdgeClassifier(x_tr.size(1), eattr.size(1)).to(dev)
    pos = float(y[tr_idx].sum())
    pos_weight = torch.tensor([(len(tr_idx) - pos) / max(pos, 1.0)], dtype=torch.float32).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val, best_state, patience = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(x_tr, ei_tr, pairs, eattr)
        loss = F.binary_cross_entropy_with_logits(logits[tr_t], yt[tr_t], pos_weight=pos_weight)
        loss.backward()
        opt.step()
        if ep % 5 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                sv = torch.sigmoid(model(x_tr, ei_tr, pairs, eattr))[val_t].cpu().numpy()
            score = average_precision_score(y[val_idx], sv) if y[val_idx].sum() else 0.0
            if score > best_val:
                best_val = score
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 6:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        # validation scored on the inner-train graph (val edges excluded)
        val_scores = torch.sigmoid(model(x_tr, ei_tr, pairs, eattr))[val_t].cpu().numpy()
        # test scored on the full-train graph (all edges strictly in test's past)
        test_t = torch.tensor(test_idx).to(dev)
        test_scores = torch.sigmoid(model(x_all, ei_all, pairs, eattr))[test_t].cpu().numpy()
    return val_scores, test_scores
