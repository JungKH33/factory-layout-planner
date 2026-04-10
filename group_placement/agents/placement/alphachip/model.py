# coding=utf-8
# Copyright 2026.
#
# AlphaChip model (network only; env-agnostic).
"""AlphaChip model: encoder + embedding + policy/value networks.

NOTE:
- This module does NOT know about FactoryLayoutEnv.
- Mask generation happens in a wrapper; this model only applies a provided mask.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.utils import scatter

from .gnn import Encoder, GraphEmbedding, PolicyNetwork, ValueNetwork


class AlphaChip(nn.Module):
  """Shared encoder + policy/value networks."""

  def __init__(
      self,
      metadata_dim: int = 12,
      node_feature_dim: int = 8,
      max_grid_size: int = 128,
      num_gcn_layers: int = 3,
      edge_fc_layers: int = 1,
      gcn_node_dim: int = 8,
      include_min_max_var: bool = True,
      is_augmented: bool = False,
      device: Optional[torch.device] = None,
      **kwargs,
  ):
    super().__init__()
    self._max_grid_size = max_grid_size
    self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    self.encoder = Encoder(
        num_gcn_layers=num_gcn_layers,
        edge_fc_layers=edge_fc_layers,
        gcn_node_dim=gcn_node_dim,
        metadata_dim=metadata_dim,
        node_feature_dim=node_feature_dim,
    )
    self.embedder = GraphEmbedding(
        gcn_node_dim=gcn_node_dim,
        include_min_max_var=include_min_max_var,
        is_augmented=is_augmented,
    )
    self.policy = PolicyNetwork(
        max_grid_size=max_grid_size,
        **kwargs,
    )
    self.value = ValueNetwork()
    self.to(self.device)

  def forward(
      self,
      data: Data | Batch,
      *,
      mask_flat: torch.Tensor,
      is_eval: bool = False,
  ):
    """Forward pass.

    - Policy returns masked flat logits [B, G*G] (masking is applied inside PolicyNetwork).
    """
    if isinstance(data, Data):
      data = Batch.from_data_list([data])
    data = data.to(self.device)

    enc = self.encoder(data)
    h = self.embedder(enc, data)

    mask_flat = mask_flat.to(device=h.device)
    logits_flat = self.policy(h, mask=mask_flat, is_eval=is_eval)  # [B,G*G]
    value = self.value(h)
    return logits_flat, value


if __name__ == "__main__":
  torch.manual_seed(0)
  metadata_dim = 12
  grid = 32

  model = AlphaChip(
      metadata_dim=metadata_dim,
      max_grid_size=grid,
  )

  macro_features1 = torch.rand(5, 8)
  netlist_graph1 = {
      "edge_index": torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
      "edge_attr": torch.rand(4, 1),
  }
  current_macro_id1 = torch.tensor([2])
  netlist_metadata1 = torch.rand(1, metadata_dim)
  mask1 = torch.ones(grid * grid, dtype=torch.int32)

  data1 = Data(
      x=macro_features1,
      edge_index=netlist_graph1["edge_index"],
      edge_attr=netlist_graph1["edge_attr"],
      netlist_metadata=netlist_metadata1,
      current_node=current_macro_id1,
  )

  macro_features2 = torch.rand(4, 8)
  netlist_graph2 = {
      "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]]),
      "edge_attr": torch.rand(3, 1),
  }
  current_macro_id2 = torch.tensor([1])
  netlist_metadata2 = torch.rand(1, metadata_dim)
  mask2 = torch.ones(grid * grid, dtype=torch.int32)

  data2 = Data(
      x=macro_features2,
      edge_index=netlist_graph2["edge_index"],
      edge_attr=netlist_graph2["edge_attr"],
      netlist_metadata=netlist_metadata2,
      current_node=current_macro_id2,
  )

  batch = Batch.from_data_list([data1, data2])
  batch = batch.to(model.device)
  print("Input macro_features shape:", tuple(macro_features1.shape), tuple(macro_features2.shape))
  print("Input edge_index shape:", tuple(netlist_graph1["edge_index"].shape), tuple(netlist_graph2["edge_index"].shape))
  print("Input edge_attr shape:", tuple(netlist_graph1["edge_attr"].shape), tuple(netlist_graph2["edge_attr"].shape))
  print("Input current_macro_id shape:", tuple(current_macro_id1.shape), tuple(current_macro_id2.shape))
  print("Input netlist_metadata shape:", tuple(netlist_metadata1.shape), tuple(netlist_metadata2.shape))
  print("Input mask shape:", tuple(mask1.shape), tuple(mask2.shape))

  with torch.no_grad():
    enc = model.encoder(batch)
    h_edges = enc["h_edges"]
    edge_batch = enc["edge_batch"]
    num_graphs = int(enc["num_graphs"].item())
    h_edges_mean = scatter(
        h_edges, edge_batch, dim=0, dim_size=num_graphs, reduce="mean"
    )
    print("netlist_metadata_embedding shape:", tuple(enc["h_metadata"].shape))
    print("macro_embedding (current node) shape:", tuple(enc["h_current_node"].shape))
    print("graph_embedding (h_edges_mean) shape:", tuple(h_edges_mean.shape))
    print("attention_embedding shape:", tuple(enc["h_attended"].shape))

  # Demonstrate final output shape with a toy mask (all-ones).
  toy_mask_flat = torch.ones(1, grid * grid, dtype=torch.int32, device=model.device)
  logits_flat, value_out = model(batch, mask_flat=toy_mask_flat, is_eval=True)
  print("GCN masked logits_flat shape:", tuple(logits_flat.shape))
  print("GCN value shape:", tuple(value_out.shape))

