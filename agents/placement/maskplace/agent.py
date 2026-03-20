from typing import Optional

import torch

from envs.action_space import ActionSpace as CandidateSet
from .model import MaskPlaceModel


class MaskPlaceAgent:
    """Pipeline-compatible agent wrapper for MaskPlace.

    External code should NOT instantiate resnet/cnn/coarse modules; this class owns a MaskPlaceModel.
    """

    def __init__(
        self,
        *,
        device: Optional[torch.device] = None,
        grid: int = 224,
        soft_coefficient: float = 1.0,
        checkpoint_path: Optional[str] = None,
        model: Optional[MaskPlaceModel] = None,
    ) -> None:
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = model or MaskPlaceModel(grid=int(grid), device=self.device, soft_coefficient=float(soft_coefficient)).to(self.device)
        if checkpoint_path is not None:
            obj = torch.load(str(checkpoint_path), map_location=self.device)
            state = obj.get("model") if isinstance(obj, dict) else obj
            if not isinstance(state, dict):
                raise ValueError("MaskPlaceAgent: checkpoint must be a state_dict or dict with key 'model'.")
            self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def policy(self, *, obs: dict, action_space: CandidateSet) -> torch.Tensor:
        state = obs.get("state", None)
        if not isinstance(state, torch.Tensor):
            raise ValueError("MaskPlaceAgent requires obs['state'].")
        st = state.to(device=self.device, dtype=torch.float32)
        if st.dim() == 1:
            st = st.view(1, -1)

        probs, _value = self.model(st)
        device = action_space.poses.device
        pri = probs[0].to(device=device, dtype=torch.float32).view(-1)

        mask = action_space.mask.to(device=device, dtype=torch.bool).view(-1)
        if int(mask.numel()) != int(pri.numel()):
            # last resort: no masking
            mask = torch.ones_like(pri, dtype=torch.bool, device=device)

        pri = pri.masked_fill(~mask, 0.0)
        s = float(pri.sum().item())
        if s > 0.0:
            pri = pri / s
        return pri

    @torch.no_grad()
    def select_action(self, *, obs: dict, action_space: CandidateSet) -> int:
        pri = self.policy(obs=obs, action_space=action_space)
        if pri.numel() == 0:
            return 0
        return int(torch.argmax(pri).item())

    @torch.no_grad()
    def value(self, *, obs: dict, action_space: CandidateSet) -> float:
        state = obs.get("state", None)
        if not isinstance(state, torch.Tensor):
            return 0.0
        st = state.to(device=self.device, dtype=torch.float32)
        if st.dim() == 1:
            st = st.view(1, -1)
        _probs, v = self.model(st)
        return float(v.view(-1)[0].item()) if isinstance(v, torch.Tensor) and v.numel() > 0 else 0.0
