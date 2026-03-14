from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from agents.base import Agent
from envs.action import EnvAction
from envs.env import FactoryLayoutEnv
from decision_adapters.base import BaseDecisionAdapter
from ordering_agents.base import OrderingAgent
from search.base import BaseSearch


@dataclass(frozen=True)
class DecisionPipeline:
    """Decision-only pipeline.

    Flow:
    engine state -> adapter.build_observation -> adapter.build_action_space
        -> agent/search select action index
        -> adapter.decode_action -> EnvAction
    """

    agent: Agent
    adapter: BaseDecisionAdapter
    search: Optional[BaseSearch] = None
    ordering_agent: Optional[OrderingAgent] = None

    def bind(self, *, engine: FactoryLayoutEnv) -> None:
        """Bind runtime engine context to adapter/search."""
        self.adapter.bind(engine)
        if self.search is not None:
            self.search.set_adapter(self.adapter)

    def decide(self) -> tuple[EnvAction, Dict[str, Any]]:
        """Return selected action and debug dictionary."""
        adapter = self.adapter

        if self.ordering_agent is not None:
            self.ordering_agent.reorder(env=adapter.engine, obs={})

        observation = adapter.build_observation()
        action_space = adapter.build_action_space()
        valid_count = adapter.num_valid_actions(action_space)
        if valid_count <= 0:
            raise ValueError("no_valid_actions")

        if self.search is None:
            action_index = int(self.agent.select_action(obs=observation, action_space=action_space))
            search_name = "none"
        else:
            action_index = int(
                self.search.select(
                    obs=observation,
                    agent=self.agent,
                    root_action_space=action_space,
                )
            )
            search_name = type(self.search).__name__

        action = adapter.decode_action(action_index, action_space)

        scores = self.agent.policy(obs=observation, action_space=action_space)
        value = self.agent.value(obs=observation, action_space=action_space)
        debug = {
            "action_index": int(action_index),
            "search": search_name,
            "valid_actions": int(valid_count),
            "action_space": action_space,
            "scores": scores.detach().to(device="cpu").numpy(),
            "value": float(value),
        }
        return action, debug


if __name__ == "__main__":
    import time
    import torch

    from envs.env_loader import load_env
    from agents.greedy import GreedyAgent
    from search.mcts import MCTSConfig, MCTSSearch
    from decision_adapters.greedy import GreedyDecisionAdapter

    ENV_JSON = "envs/env_configs/basic_01.json"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    loaded = load_env(ENV_JSON, device=device)
    engine = loaded.env
    engine.log = False
    adapter = GreedyDecisionAdapter(k=50, scan_step=10.0, quant_step=10.0, random_seed=0)
    search = MCTSSearch(config=MCTSConfig(num_simulations=50, rollout_enabled=True, rollout_depth=5))

    agent = GreedyAgent(prior_temperature=1.0)
    pipe = DecisionPipeline(agent=agent, adapter=adapter, search=search)
    pipe.bind(engine=engine)
    _obs_env, _info = engine.reset(options=loaded.reset_kwargs)

    t0 = time.perf_counter()
    try:
        action, dbg = pipe.decide()
        _obs_env2, reward, terminated, truncated, info = engine.step_action(action)
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
    print(" result=", {"reason": reason, "action": None if reason != "ok" else (action.x, action.y, action.rot)})
    print(" debug=", dbg)
    print(" reward=", reward, "terminated=", terminated, "truncated=", truncated, "reason=", info.get("reason"))
    print(f" elapsed_ms={dt_ms:.2f}")
