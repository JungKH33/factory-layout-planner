# coding=utf-8
# Copyright 2026.
#
# PyTorch Geometric implementation of AlphaChip-style GCN.
"""Edge-GNN encoder + policy/value heads (GCN path)."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.utils import scatter, softmax

DEFAULT_METADATA_DIM = 12
DEFAULT_MAX_GRID_SIZE = 128
DEFAULT_NODE_FEATURE_DIM = 8


def _mask_logits(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
  """Apply action mask on flat logits.

  - logits: [B, A]
  - mask:  [B, A] or [A] (broadcasted)
  """
  if mask is None:
    return logits
  if mask.dim() < logits.dim():
    mask = mask.unsqueeze(-2)
  almost_neg_inf = torch.ones_like(logits) * (-(2.0**32) + 1)
  return torch.where(mask.bool(), logits, almost_neg_inf)


class Encoder(nn.Module):
  """Edge-GNN encoder."""

  def __init__(
      self,
      num_gcn_layers: int = 3,
      edge_fc_layers: int = 1,
      gcn_node_dim: int = 8,
      metadata_dim: int = DEFAULT_METADATA_DIM,
      node_feature_dim: int = DEFAULT_NODE_FEATURE_DIM,
  ):
    super().__init__()
    self._num_gcn_layers = num_gcn_layers
    self._gcn_node_dim = gcn_node_dim

    self._metadata_encoder = nn.Sequential(
        nn.Linear(metadata_dim, self._gcn_node_dim),
        nn.ReLU(),
    )
    self._feature_encoder = nn.Sequential(
        nn.Linear(node_feature_dim, self._gcn_node_dim),
        nn.ReLU(),
    )

    self._edge_fc_list = nn.ModuleList(
        [
            self._create_edge_fc(edge_fc_layers)
            for _ in range(self._num_gcn_layers)
        ]
    )

    self._attention_query_layer = nn.Linear(self._gcn_node_dim, self._gcn_node_dim)
    self._attention_key_layer = nn.Linear(self._gcn_node_dim, self._gcn_node_dim)
    self._attention_value_layer = nn.Linear(self._gcn_node_dim, self._gcn_node_dim)

  def _create_edge_fc(self, edge_fc_layers: int) -> nn.Sequential:
    layers = []
    input_dim = self._gcn_node_dim * 2 + 1
    for _ in range(edge_fc_layers):
      layers.append(nn.Linear(input_dim, self._gcn_node_dim))
      layers.append(nn.ReLU())
      input_dim = self._gcn_node_dim
    return nn.Sequential(*layers)

  def _edge_to_node_mean(
      self, h_edges: torch.Tensor, edge_index: torch.Tensor, num_nodes: int
  ) -> torch.Tensor:
    src = edge_index[0]
    dst = edge_index[1]
    node_index = torch.cat([src, dst], dim=0)
    edge_repeated = torch.cat([h_edges, h_edges], dim=0)
    return scatter(edge_repeated, node_index, dim=0, dim_size=num_nodes, reduce="mean")

  def _self_attention(
      self,
      h_current_node: torch.Tensor,
      h_nodes: torch.Tensor,
      batch: torch.Tensor,
  ) -> torch.Tensor:
    query = self._attention_query_layer(h_current_node)
    keys = self._attention_key_layer(h_nodes)
    values = self._attention_value_layer(h_nodes)

    query_per_node = query[batch]
    scale = math.sqrt(query.shape[-1])
    scores = (keys * query_per_node).sum(dim=-1) / scale
    attn_weights = softmax(scores, batch)
    attended = scatter(
        attn_weights.unsqueeze(-1) * values, batch, dim=0, reduce="sum"
    )
    return attended

  def forward(self, data: Data | Batch) -> dict[str, torch.Tensor]:
    if isinstance(data, Data):
      data = Batch.from_data_list([data])

    x = data.x
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    if edge_attr is None:
      raise ValueError("edge_attr is required and must be shape [E, 1].")
    if edge_attr.dim() == 1:
      edge_attr = edge_attr.unsqueeze(-1)

    batch = data.batch
    num_nodes = x.shape[0]
    num_graphs = int(batch.max().item()) + 1

    netlist_metadata = data.netlist_metadata
    if netlist_metadata.dim() == 1:
      netlist_metadata = netlist_metadata.unsqueeze(0)
    h_metadata = self._metadata_encoder(netlist_metadata)

    h_nodes = self._feature_encoder(x)

    mask = edge_attr.ne(0.0).expand(-1, self._gcn_node_dim * 2 + 1)
    h_edges_i_j = torch.cat(
        [h_nodes[edge_index[0]], h_nodes[edge_index[1]], edge_attr], dim=-1
    )
    h_edges_j_i = torch.cat(
        [h_nodes[edge_index[1]], h_nodes[edge_index[0]], edge_attr], dim=-1
    )
    h_edges_i_j = torch.where(mask, h_edges_i_j, torch.zeros_like(h_edges_i_j))
    h_edges_j_i = torch.where(mask, h_edges_j_i, torch.zeros_like(h_edges_j_i))

    for i in range(self._num_gcn_layers):
      h_edges = (
          self._edge_fc_list[i](h_edges_i_j)
          + self._edge_fc_list[i](h_edges_j_i)
      ) / 2.0
      h_nodes_new = self._edge_to_node_mean(h_edges, edge_index, num_nodes)
      h_nodes = h_nodes_new + h_nodes

    edge_batch = batch[edge_index[0]]

    current_node = data.current_node
    if current_node.dim() > 1:
      current_node = current_node.squeeze(-1)
    if hasattr(data, "ptr"):
      offsets = data.ptr[:-1].to(current_node.device)
      current_node = current_node + offsets
    h_current_node = h_nodes[current_node]

    h_attended = self._self_attention(h_current_node, h_nodes, batch)

    return {
        "h_metadata": h_metadata,
        "h_edges": h_edges,
        "edge_batch": edge_batch,
        "num_graphs": torch.tensor(num_graphs, device=h_nodes.device),
        "h_current_node": h_current_node,
        "h_attended": h_attended,
    }


class GraphEmbedding(nn.Module):
  """Builds graph embedding h from encoder outputs."""

  def __init__(
      self,
      gcn_node_dim: int = 8,
      include_min_max_var: bool = True,
      is_augmented: bool = False,
  ):
    super().__init__()
    self._include_min_max_var = include_min_max_var
    self._is_augmented = is_augmented
    if self._is_augmented:
      self._augmented_embedding_layer = nn.LazyLinear(gcn_node_dim)

  def forward(
      self,
      enc: dict[str, torch.Tensor],
      data: Data | Batch,
      finetune_value_only: bool = False,
  ) -> torch.Tensor:
    h_edges = enc["h_edges"]
    edge_batch = enc["edge_batch"]
    num_graphs = int(enc["num_graphs"].item())

    h_edges_mean = scatter(h_edges, edge_batch, dim=0, dim_size=num_graphs, reduce="mean")
    observation_hiddens = [enc["h_metadata"], h_edges_mean]
    if self._include_min_max_var:
      h_edges_max = scatter(h_edges, edge_batch, dim=0, dim_size=num_graphs, reduce="max")
      h_edges_min = scatter(h_edges, edge_batch, dim=0, dim_size=num_graphs, reduce="min")
      h_edges_mean_for_var = h_edges_mean[edge_batch]
      var_num = scatter(
          (h_edges - h_edges_mean_for_var).pow(2),
          edge_batch,
          dim=0,
          dim_size=num_graphs,
          reduce="mean",
      )
      observation_hiddens.extend([var_num, h_edges_max, h_edges_min])

    observation_hiddens.append(enc["h_attended"])
    observation_hiddens.append(enc["h_current_node"])

    if self._is_augmented and hasattr(data, "augmented_features"):
      augmented_embedding = self._augmented_embedding_layer(data.augmented_features)
      observation_hiddens.append(augmented_embedding)

    h = torch.cat(observation_hiddens, dim=1)
    if finetune_value_only:
      h = h.detach()
    return h


class PolicyNetwork(nn.Module):
  """Policy network with deconv head."""

  EPSILON = 1e-6

  def __init__(
      self,
      max_grid_size: int = DEFAULT_MAX_GRID_SIZE,
      dirichlet_alpha: float = 0.1,
      policy_noise_weight: float = 0.0,
  ):
    super().__init__()
    self._dirichlet_alpha = dirichlet_alpha
    self._policy_noise_weight = policy_noise_weight

    grid = max_grid_size
    self._policy_location_head = nn.Sequential(
        nn.LazyLinear((grid // 16) * (grid // 16) * 32),
        nn.ReLU(),
        _ReshapeLayer((-1, 32, grid // 16, grid // 16)),
        nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(4, 2, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(2, 1, kernel_size=3, stride=1, padding=1),
        nn.Flatten(),
    )

  def _add_noise(self, logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    alphas = torch.full_like(probs, self._dirichlet_alpha)
    noise = torch.distributions.Dirichlet(alphas).sample()
    noised_probs = (1.0 - self._policy_noise_weight) * probs + (
        self._policy_noise_weight
    ) * noise
    return torch.log(noised_probs + self.EPSILON)

  def forward(
      self,
      h: torch.Tensor,
      mask: Optional[torch.Tensor] = None,
      is_eval: bool = False,
  ) -> torch.Tensor:
    """Return masked flat logits [B, G*G]."""
    location_logits = self._policy_location_head(h)  # [B,G*G]
    logits = location_logits if is_eval else self._add_noise(location_logits)
    return _mask_logits(logits, mask)


class ValueNetwork(nn.Module):
  """Value network MLP."""

  def __init__(self):
    super().__init__()
    self._value_head = nn.Sequential(
        nn.LazyLinear(32),
        nn.ReLU(),
        nn.Linear(32, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )

  def forward(self, h: torch.Tensor) -> torch.Tensor:
    value = self._value_head(h)
    return value.squeeze(-1)


class _ReshapeLayer(nn.Module):
  def __init__(self, shape):
    super().__init__()
    self._shape = shape

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.view(x.shape[0], *self._shape[1:])


if __name__ == "__main__":
  torch.manual_seed(0)
  grid = 32
  metadata_dim = DEFAULT_METADATA_DIM

  encoder = Encoder(metadata_dim=metadata_dim)
  embedder = GraphEmbedding()
  policy = PolicyNetwork(max_grid_size=grid)
  value = ValueNetwork()

  macro_features = torch.rand(5, DEFAULT_NODE_FEATURE_DIM)
  netlist_graph = {
      "edge_index": torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
      "edge_attr": torch.rand(4, 1),
  }
  current_macro_id = torch.tensor([2])
  netlist_metadata = torch.rand(1, metadata_dim)
  mask = torch.ones(grid * grid, dtype=torch.int32)

  data = Data(
      x=macro_features,
      edge_index=netlist_graph["edge_index"],
      edge_attr=netlist_graph["edge_attr"],
      netlist_metadata=netlist_metadata,
      current_node=current_macro_id,
  )

  enc = encoder(data)
  h = embedder(enc, data)
  mask = torch.ones(1, grid * grid, dtype=torch.int32)
  logits = policy(h, mask=mask, is_eval=True)
  value_out = value(h)
  print("GCN policy logits shape:", tuple(logits.shape))
  print("GCN value shape:", tuple(value_out.shape))
