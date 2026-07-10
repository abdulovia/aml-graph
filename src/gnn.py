"""GraphSAGE edge classifier for AML with light Multi-GNN-style tricks.

Design choices that keep the split leakage-free:

* Message passing uses **only training edges** to build node embeddings, so a
  test edge can never inject future information into its endpoints' vectors.
* Reverse edges are added (Multi-GNN "reverse message passing") so a receiver
  account can aggregate signal from its senders.
* A per-edge motif feature vector is concatenated at the classification head,
  fusing the interpretable detectors with the learned representation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import SAGEConv

from src.config import SEED


@dataclass
class GraphTensors:
    """Tensors describing the graph for the GNN.

    Attributes:
        x: Node feature matrix ``(n_nodes, n_node_feats)``.
        train_edge_index: ``(2, E_train*2)`` incl. reverse edges, for message
            passing.
        edge_pairs: ``(2, E)`` endpoint indices for every edge to classify.
        edge_attr: ``(E, n_edge_feats)`` motif features per edge.
        y: ``(E,)`` labels.
        train_mask/test_mask: Boolean masks over the ``E`` classified edges.

    """

    x: Tensor
    train_edge_index: Tensor
    edge_pairs: Tensor
    edge_attr: Tensor
    y: Tensor
    train_mask: Tensor
    test_mask: Tensor


def build_graph_tensors(
    src: list[str],
    dst: list[str],
    edge_attr: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> GraphTensors:
    """Assemble PyG tensors with train-only message-passing edges.

    Node features are simple degree/volume statistics computed from **training
    edges only** (again, to prevent leakage).
    """
    nodes = sorted(set(src) | set(dst))
    idx_of = {n: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)

    su = np.array([idx_of[s] for s in src], dtype=np.int64)
    dv = np.array([idx_of[d] for d in dst], dtype=np.int64)

    # Node features from training edges only (leakage-safe): simple degree/volume
    # statistics. The interpretable motif features are fused at the edge-scoring
    # head instead of the node level, which trained more stably here.
    in_deg = np.zeros(n_nodes, dtype=np.float32)
    out_deg = np.zeros(n_nodes, dtype=np.float32)
    for i in train_idx:
        out_deg[su[i]] += 1
        in_deg[dv[i]] += 1
    x = np.stack([np.log1p(in_deg), np.log1p(out_deg), (in_deg > 0).astype(np.float32)], axis=1)

    fwd = np.stack([su[train_idx], dv[train_idx]])
    rev = np.stack([dv[train_idx], su[train_idx]])
    train_edge_index = np.concatenate([fwd, rev], axis=1)

    edge_pairs = np.stack([su, dv])
    train_mask = np.zeros(y.size, dtype=bool)
    test_mask = np.zeros(y.size, dtype=bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    return GraphTensors(
        x=torch.tensor(x, dtype=torch.float32),
        train_edge_index=torch.tensor(train_edge_index, dtype=torch.long),
        edge_pairs=torch.tensor(edge_pairs, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.float32),
        train_mask=torch.tensor(train_mask),
        test_mask=torch.tensor(test_mask),
    )


class SAGEEdgeClassifier(nn.Module):
    """Two-layer GraphSAGE encoder + MLP edge-scoring head."""

    def __init__(self, n_node_feats: int, n_edge_feats: int, hidden: int = 64) -> None:
        """Initialise encoder and head.

        Args:
            n_node_feats: Node feature dimension.
            n_edge_feats: Per-edge motif feature dimension.
            hidden: Hidden width.

        """
        super().__init__()
        self.conv1 = SAGEConv(n_node_feats, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + n_edge_feats, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Compute node embeddings via message passing."""
        h = F.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        return h

    def forward(self, gt: GraphTensors) -> Tensor:
        """Return per-edge logits for every classified edge."""
        h = self.encode(gt.x, gt.train_edge_index)
        hu = h[gt.edge_pairs[0]]
        hv = h[gt.edge_pairs[1]]
        z = torch.cat([hu, hv, gt.edge_attr], dim=1)
        return self.head(z).squeeze(-1)


def train_gnn(
    gt: GraphTensors, epochs: int = 60, lr: float = 5e-3, device: str = "cpu"
) -> np.ndarray:
    """Train the edge classifier and return test-edge positive scores.

    Args:
        gt: Graph tensors from :func:`build_graph_tensors`.
        epochs: Training epochs.
        lr: Learning rate.
        device: ``"cpu"``, ``"mps"`` or ``"cuda"``.

    Returns:
        Sigmoid scores for the test edges, aligned with ``test_mask`` order.

    """
    torch.manual_seed(SEED)
    torch.set_num_threads(2)  # cap threads: default over-subscribes CPU under Rosetta
    dev = torch.device(device)
    model = SAGEEdgeClassifier(gt.x.size(1), gt.edge_attr.size(1)).to(dev)
    gt = GraphTensors(
        gt.x.to(dev),
        gt.train_edge_index.to(dev),
        gt.edge_pairs.to(dev),
        gt.edge_attr.to(dev),
        gt.y.to(dev),
        gt.train_mask.to(dev),
        gt.test_mask.to(dev),
    )
    pos = gt.y[gt.train_mask].sum().clamp(min=1.0)
    neg = (~gt.y[gt.train_mask].bool()).sum().clamp(min=1.0)
    pos_weight = (neg / pos).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(gt)
        loss = F.binary_cross_entropy_with_logits(
            logits[gt.train_mask], gt.y[gt.train_mask], pos_weight=pos_weight
        )
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(gt))[gt.test_mask].cpu().numpy()
    return scores
