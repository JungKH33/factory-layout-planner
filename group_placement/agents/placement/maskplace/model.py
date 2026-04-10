from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class StateLayout:
    """
    1D state vector layout definition.

    Assumes:
      state = [pos_idx] + [grid*grid*5 maps] + [extra(2)]
    maps order (based on your usage):
      map0: x[:, 1 : 1+g2]                        # used in coarse_input
      map1: x[:, 1+g2 : 1+2*g2]                   # net_img base
      map2: x[:, 1+2*g2 : 1+3*g2]                 # mask (hard constraint) + net_img penalty add
      map3: x[:, 1+3*g2 : 1+4*g2]                 # used in coarse_input
      map4: x[:, 1+4*g2 : 1+5*g2]                 # cnn_input 4ch includes map1..map4? (your code uses 4 maps: map1..map4)
    """
    grid: int

    def __post_init__(self):
        g2 = self.grid * self.grid
        object.__setattr__(self, "g2", g2)

    @property
    def pos_idx_slice(self) -> slice:
        return slice(0, 1)

    def map_slice(self, k: int) -> slice:
        # k in [0..4]
        start = 1 + self.g2 * k
        end = 1 + self.g2 * (k + 1)
        return slice(start, end)

    @property
    def maps_slice(self) -> slice:
        return slice(1, 1 + self.g2 * 5)


class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 8, 1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn(x)


class MyCNNCoarse(nn.Module):
    def __init__(self, res_net: nn.Module, device: torch.device):
        super().__init__()
        self.cnn = res_net.to(device)
        self.cnn.fc = nn.Linear(512, 16 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),  # 14
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding=1),   # 28
            nn.ReLU(),
            nn.ConvTranspose2d(4, 2, 3, stride=2, padding=1, output_padding=1),   # 56
            nn.ReLU(),
            nn.ConvTranspose2d(2, 1, 3, stride=2, padding=1, output_padding=1),   # 112
            nn.ReLU(),
            nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1, output_padding=1),   # 224
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x).reshape(-1, 16, 7, 7)
        return self.deconv(x)


class Actor(nn.Module):
    def __init__(
        self,
        layout: StateLayout,
        cnn: nn.Module,
        cnn_coarse: nn.Module,
        soft_coefficient: float = 1.0,
    ):
        super().__init__()
        self.layout = layout
        self.grid = layout.grid
        self.g2 = layout.grid * layout.grid
        self.soft_coefficient = float(soft_coefficient)

        self.cnn = cnn
        self.cnn_coarse = cnn_coarse

        self.softmax = nn.Softmax(dim=-1)
        self.merge = nn.Conv2d(2, 1, 1)

    def forward(
        self,
        state: torch.Tensor,
        cnn_res: Optional[torch.Tensor] = None,
        gcn_res: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
          action_probs: (B, grid*grid)
          cnn_res: (B,1,grid,grid)
          gcn_res: passthrough (unused now)
        """
        B = state.shape[0]
        g = self.grid
        g2 = self.g2

        # Build CNN feature if not provided
        if cnn_res is None:
            # cnn_input: 4ch from maps 1..4 (same as your original slicing)
            cnn_input = state[:, self.layout.map_slice(1)]  # map1
            cnn_input = torch.cat(
                [
                    cnn_input,
                    state[:, self.layout.map_slice(2)],  # map2
                    state[:, self.layout.map_slice(3)],  # map3
                    state[:, self.layout.map_slice(4)],  # map4
                ],
                dim=1,
            ).reshape(B, 4, g, g)

            # hard mask from map2
            mask = state[:, self.layout.map_slice(2)].reshape(B, g, g).flatten(1)

            cnn_res_fine = self.cnn(cnn_input)  # (B,1,g,g)

            # coarse input: cat(map0,map1,map3) -> 3ch
            coarse_input = torch.cat(
                [
                    state[:, self.layout.map_slice(0)],  # map0
                    state[:, self.layout.map_slice(1)],  # map1
                    state[:, self.layout.map_slice(3)],  # map3
                ],
                dim=1,
            ).reshape(B, 3, g, g)

            cnn_res_coarse = self.cnn_coarse(coarse_input)  # (B,1,g,g)
            cnn_res = self.merge(torch.cat((cnn_res_fine, cnn_res_coarse), dim=1))
        else:
            # if provided, still need mask for action masking below
            mask = state[:, self.layout.map_slice(2)].reshape(B, g, g).flatten(1)

        # net_img logic (your original):
        net_img = state[:, self.layout.map_slice(1)]  # map1
        net_img = net_img + state[:, self.layout.map_slice(2)] * 10.0  # map2 penalty
        net_img_min = net_img.min() + float(self.soft_coefficient)
        mask2 = net_img.le(net_img_min).logical_not().float()  # 1 where NOT minimal

        logits = cnn_res.reshape(B, g2)
        logits = torch.where((mask + mask2) >= 1.0, torch.tensor(-1.0e10, device=logits.device, dtype=logits.dtype), logits)
        probs = self.softmax(logits)

        return probs, cnn_res, gcn_res


class Critic(nn.Module):
    def __init__(self, pos_vocab: int = 1400):
        super().__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.state_value = nn.Linear(64, 1)
        self.pos_emb = nn.Embedding(pos_vocab, 64)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        pos_idx = state[:, 0].long()
        x1 = F.relu(self.fc1(self.pos_emb(pos_idx)))
        x2 = F.relu(self.fc2(x1))
        return self.state_value(x2)


class MaskPlaceModel(nn.Module):
    """Single nn.Module container for MaskPlace.

    - Owns pretrained ResNet18 backbone, fine CNN, Actor, and Critic.
    - Enables checkpoint save/load in the style: {"model": model.state_dict()}.

    Notes:
    - We intentionally do NOT expose a `pretrained_backbone` knob here. MaskPlace uses pretrained ResNet18.
    """

    def __init__(
        self,
        *,
        grid: int = 224,
        device: Optional[torch.device] = None,
        soft_coefficient: float = 1.0,
    ) -> None:
        super().__init__()
        self.grid = int(grid)
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # Build submodules (ResNet is created here; external code should not instantiate it).
        import torchvision

        layout = StateLayout(grid=int(self.grid))
        cnn_fine = MyCNN().to(self.device)
        resnet = torchvision.models.resnet18(pretrained=True)
        cnn_coarse = MyCNNCoarse(res_net=resnet, device=self.device).to(self.device)

        self.actor = Actor(layout=layout, cnn=cnn_fine, cnn_coarse=cnn_coarse, soft_coefficient=float(soft_coefficient)).to(self.device)
        self.critic = Critic().to(self.device)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (action_probs[B,A], value[B,1])."""
        probs, _cnn_res, _ = self.actor(state)
        value = self.critic(state)
        return probs, value
