from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool


class ArchitectureGNN(nn.Module):
    """
    3-layer Graph Attention Network that analyzes LM architecture graphs.

    Runs in FP32. Returns per-node embeddings (used by FiLM) and a
    graph-level embedding (used for logging / analysis).

    Architecture:
        Layer 1: GATConv(16  → 64, heads=4, concat) → 256-dim
        Layer 2: GATConv(256 → 64, heads=4, concat) → 256-dim
        Layer 3: GATConv(256 → 64, heads=1, no concat) → 64-dim
        global_mean_pool → graph embedding [1, 64]
    """

    def __init__(
        self,
        in_channels: int = 16,
        hidden: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv1 = GATConv(
            in_channels, hidden, heads=heads, concat=True, dropout=dropout
        )
        self.conv2 = GATConv(
            hidden * heads, hidden, heads=heads, concat=True, dropout=dropout
        )
        self.conv3 = GATConv(
            hidden * heads, hidden, heads=1, concat=False, dropout=dropout
        )
        self.act = nn.ELU()
        self.drop = nn.Dropout(dropout)

    def forward(
        self, data: Data
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            data: torch_geometric Data with x=[N, 16] and edge_index=[2, E].

        Returns:
            node_embeddings: [N, 64]
            graph_embedding:  [1, 64]
        """
        x, edge_index = data.x.float(), data.edge_index

        # batch vector: all zeros for a single graph
        batch: torch.Tensor
        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.act(self.conv1(x, edge_index))
        x = self.drop(x)
        x = self.act(self.conv2(x, edge_index))
        x = self.drop(x)
        x = self.conv3(x, edge_index)  # [N, 64]

        node_embeddings = x
        graph_embedding = global_mean_pool(x, batch)  # [1, 64]

        return node_embeddings, graph_embedding
