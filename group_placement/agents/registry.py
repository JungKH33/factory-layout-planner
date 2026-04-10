"""Agent + Adapter registry.

Maps method names to valid (agent, adapter) combinations with factory functions.

Usage::

    from agents.registry import create

    agent, adapter = create(
        method="greedyv3",
        adapter_kwargs={"k": 50, "quant_step": 10.0},
    )

    # Override default agent for a method:
    agent, adapter = create(
        method="maskplace",
        agent="greedy",
        agent_kwargs={"prior_temperature": 1.0},
        adapter_kwargs={"grid": 224},
    )
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from .base import Agent, BaseAdapter
from .placement.greedy import (
    GreedyAgent,
    GreedyAdapter,
    GreedyV2Adapter,
    GreedyV3Adapter,
    GreedyV4Adapter,
    GreedyV5Adapter,
)
from .placement.maskplace import MaskPlaceAgent, MaskPlaceAdapter
from .placement.alphachip import AlphaChipAgent, AlphaChipAdapter


@dataclass(frozen=True)
class AgentAdapterSpec:
    """A valid (agent, adapter) combination."""
    agent_factory: Callable[..., Agent]
    adapter_factory: Callable[..., BaseAdapter]


# method name -> { agent name -> spec }
REGISTRY: Dict[str, Dict[str, AgentAdapterSpec]] = {
    "greedy": {
        "greedy": AgentAdapterSpec(GreedyAgent, GreedyAdapter),
    },
    "greedyv2": {
        "greedy": AgentAdapterSpec(GreedyAgent, GreedyV2Adapter),
    },
    "greedyv3": {
        "greedy": AgentAdapterSpec(GreedyAgent, GreedyV3Adapter),
    },
    "greedyv4": {
        "greedy": AgentAdapterSpec(GreedyAgent, GreedyV4Adapter),
    },
    "greedyv5": {
        "greedy": AgentAdapterSpec(GreedyAgent, GreedyV5Adapter),
    },
    "maskplace": {
        "maskplace": AgentAdapterSpec(MaskPlaceAgent, MaskPlaceAdapter),
        "greedy": AgentAdapterSpec(GreedyAgent, MaskPlaceAdapter),
    },
    "alphachip": {
        "alphachip": AgentAdapterSpec(AlphaChipAgent, AlphaChipAdapter),
        "greedy": AgentAdapterSpec(GreedyAgent, AlphaChipAdapter),
    },
}

# method -> default agent key
DEFAULT_AGENT: Dict[str, str] = {
    "greedy": "greedy",
    "greedyv2": "greedy",
    "greedyv3": "greedy",
    "greedyv4": "greedy",
    "greedyv5": "greedy",
    "maskplace": "maskplace",
    "alphachip": "alphachip",
}


def create(
    method: str,
    *,
    agent: Optional[str] = None,
    agent_kwargs: Optional[Dict[str, Any]] = None,
    adapter_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Agent, BaseAdapter]:
    """Create an (agent, adapter) pair.

    Parameters
    ----------
    method : str
        Adapter method: ``"greedy"`` | ``"greedyv2"`` | ``"greedyv3"``
        | ``"greedyv4"`` | ``"greedyv5"``
        | ``"maskplace"`` | ``"alphachip"``.
    agent : str, optional
        Override agent type.  Defaults to the method's natural agent
        (e.g. ``"greedy"`` for greedy adapters, ``"maskplace"`` for maskplace).
    agent_kwargs : dict, optional
        Keyword arguments forwarded to the agent constructor.
    adapter_kwargs : dict, optional
        Keyword arguments forwarded to the adapter constructor.

    Returns
    -------
    (Agent, BaseAdapter)
    """
    if method not in REGISTRY:
        raise ValueError(f"Unknown method {method!r}, choices: {list(REGISTRY)}")
    agent_key = agent or DEFAULT_AGENT[method]
    specs = REGISTRY[method]
    if agent_key not in specs:
        raise ValueError(
            f"Agent {agent_key!r} is not compatible with method {method!r}. "
            f"Valid agents: {list(specs)}"
        )
    spec = specs[agent_key]
    return (
        spec.agent_factory(**(agent_kwargs or {})),
        spec.adapter_factory(**(adapter_kwargs or {})),
    )
