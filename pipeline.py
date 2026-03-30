from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from agents.base import Agent, BaseAdapter, BaseHierarchicalAdapter, OrderingAgent
from envs.env import FactoryLayoutEnv
from envs.placement.base import GroupPlacement
from search.base import BaseHierarchicalSearch, BaseSearch

# Return type: all adapters return resolved placement only.
DecideResult = Tuple[GroupPlacement, Dict[str, Any]]


@dataclass(frozen=True)
class DecisionPipeline:
    """Decision-only pipeline.

    All adapters:
        -> adapter.resolve_action -> GroupPlacement
        -> caller uses engine.step_placement()
    """

    agent: Agent
    adapter: BaseAdapter
    search: Optional[BaseSearch] = None
    ordering_agent: Optional[OrderingAgent] = None

    @property
    def is_hierarchical(self) -> bool:
        return isinstance(self.adapter, BaseHierarchicalAdapter)

    def bind(self, *, engine: FactoryLayoutEnv) -> None:
        """Bind runtime engine context to adapter/search."""
        self.adapter.bind(engine)
        if self.search is not None:
            self.search.set_adapter(self.adapter)

    def decide(self) -> DecideResult:
        """Return action result and debug dictionary."""
        adapter = self.adapter

        if self.ordering_agent is not None:
            self.ordering_agent.reorder(env=adapter.engine, obs={})

        observation = adapter.build_observation()
        action_space = adapter.build_action_space()
        valid_count = adapter.num_valid_actions(action_space)
        if valid_count <= 0:
            raise ValueError("no_valid_actions")

        if isinstance(self.search, BaseHierarchicalSearch):
            if not isinstance(adapter, BaseHierarchicalAdapter):
                raise TypeError(
                    f"{type(self.search).__name__} requires BaseHierarchicalAdapter, "
                    f"got {type(adapter).__name__}"
                )
            cell_idx, local_idx = self.search.select_h(
                obs=observation,
                agent=self.agent,
                root_action_space=action_space,
            )
            worker_as = adapter.cell_action_space(cell_idx)
            placement = adapter.resolve_worker_action(
                local_idx, worker_as, cell_idx=cell_idx,
            )
            action_index = cell_idx
            search_name = type(self.search).__name__
        elif self.search is None:
            action_index = int(self.agent.select_action(obs=observation, action_space=action_space))
            search_name = "none"
            placement = adapter.resolve_action(action_index, action_space)
        else:
            action_index = int(self.search.select(
                obs=observation, agent=self.agent, root_action_space=action_space,
            ))
            search_name = type(self.search).__name__
            placement = adapter.resolve_action(action_index, action_space)

        debug = self._build_debug(action_index, search_name, valid_count, action_space, observation)
        return placement, debug

    def _build_debug(self, action_index, search_name, valid_count, action_space, observation):
        scores = self.agent.policy(obs=observation, action_space=action_space)
        value = self.agent.value(obs=observation, action_space=action_space)
        return {
            "action_index": int(action_index),
            "search": search_name,
            "valid_actions": int(valid_count),
            "action_space": action_space,
            "scores": scores.detach().to(device="cpu").numpy(),
            "value": float(value),
        }


if __name__ == "__main__":
    import time
    import torch

    from envs.env_loader import load_env
    from agents.placement.greedy import GreedyAgent, GreedyAdapter
    from search.mcts import MCTSConfig, MCTSSearch

    ENV_JSON = "envs/env_configs/basic_01.json"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    loaded = load_env(ENV_JSON, device=device)
    engine = loaded.env
    engine.log = False
    adapter = GreedyAdapter(k=50, scan_step=10.0, quant_step=10.0, random_seed=0)
    search = MCTSSearch(config=MCTSConfig(num_simulations=50, rollout_enabled=True, rollout_depth=5))

    agent = GreedyAgent(prior_temperature=1.0)
    pipe = DecisionPipeline(agent=agent, adapter=adapter, search=search)
    pipe.bind(engine=engine)
    _obs_env, _info = engine.reset(options=loaded.reset_kwargs)

    t0 = time.perf_counter()
    try:
        result, dbg = pipe.decide()
        _obs_env2, reward, terminated, truncated, info = engine.step_placement(result)
        reason = "ok"
    except ValueError as e:
        if str(e) == "no_valid_actions":
            reward = float(engine.failure_penalty())
            terminated = False
            truncated = True
            info = {"reason": "no_valid_actions"}
            reason = "no_valid_actions"
            dbg = {"reason": "no_valid_actions"}
        else:
            raise
    dt_ms = (time.perf_counter() - t0) * 1000.0

    print("pipeline demo")
    print(" env=", ENV_JSON, "device=", device, "next_gid=", (engine.get_state().remaining[0] if engine.get_state().remaining else None))
    print(" debug=", dbg)
    print(" reward=", reward, "terminated=", terminated, "truncated=", truncated, "reason=", info.get("reason"))
    print(f" elapsed_ms={dt_ms:.2f}")
