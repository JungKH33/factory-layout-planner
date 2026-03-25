"""FastAPI WebUI application for Factory Layout."""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, get_type_hints

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Thread pool for running sync search in background
_search_executor = ThreadPoolExecutor(max_workers=2)
logger = logging.getLogger(__name__)

from envs.action_space import ActionSpace
from agents.placement.greedy import GreedyAgent, GreedyAdapter, GreedyV2Adapter, GreedyV3Adapter
from agents.placement.alphachip import AlphaChipAdapter
from agents.placement.maskplace import MaskPlaceAdapter
from search.mcts import MCTSConfig
from search.beam import BeamConfig
from envs.action import EnvAction
from envs.state import EnvState


def _extract_params(cls, exclude: set = None) -> Dict[str, Dict[str, Any]]:
    """Extract parameter info from class __init__ signature or dataclass fields."""
    import dataclasses
    exclude = exclude or set()
    params = {}
    
    # Check if it's a dataclass
    if dataclasses.is_dataclass(cls):
        for field in dataclasses.fields(cls):
            name = field.name
            if name in exclude or name.startswith('_'):
                continue
            
            info = {"name": name}
            
            # Get default value and required status
            if field.default is not dataclasses.MISSING:
                info["default"] = field.default
                info["required"] = False
            elif field.default_factory is not dataclasses.MISSING:
                info["default"] = None
                info["required"] = False
            else:
                info["default"] = None
                info["required"] = True
            
            # Get type
            if field.type is not None:
                type_name = field.type.__name__ if hasattr(field.type, '__name__') else str(field.type)
                info["type"] = type_name
            elif info["default"] is not None:
                info["type"] = type(info["default"]).__name__
            else:
                info["type"] = "str"
            
            params[name] = info
        return params
    
    # Regular class - use __init__ signature
    sig = inspect.signature(cls.__init__)
    
    for name, param in sig.parameters.items():
        if name in ('self', 'engine', 'device', 'checkpoint_path') or name in exclude:
            continue
        if name.startswith('_'):
            continue
            
        info = {"name": name}
        
        # Get default value and required status
        if param.default is not inspect.Parameter.empty:
            info["default"] = param.default
            info["required"] = False
        else:
            info["default"] = None
            info["required"] = True
            
        # Infer type from default or annotation
        if param.annotation is not inspect.Parameter.empty:
            ann = param.annotation
            if hasattr(ann, '__origin__'):  # Optional, List, etc.
                ann = ann.__args__[0] if ann.__args__ else str
            info["type"] = ann.__name__ if hasattr(ann, '__name__') else str(ann)
        elif info["default"] is not None:
            info["type"] = type(info["default"]).__name__
        else:
            info["type"] = "str"
            
        params[name] = info
    
    return params


# Registry of components and their classes
WRAPPER_CLASSES = {
    "greedy": GreedyAdapter,
    "greedyv2": GreedyV2Adapter,
    "greedyv3": GreedyV3Adapter,
    "alphachip": AlphaChipAdapter,
    "maskplace": MaskPlaceAdapter,
}

SEARCH_CLASSES = {
    "mcts": MCTSConfig,
    "beam": BeamConfig,
}

AGENT_CLASSES = {
    "greedy": GreedyAgent,
}

from webui.schemas import (
    SessionCreateRequest,
    StepRequest,
    SearchRequest,
    SessionState,
    CandidateInfo,
    SearchProgress,
)
from webui.session import manager, Session


app = FastAPI(title="Factory Layout WebUI")

# Mount static files
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    """Serve the main HTML page."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/configs")
async def list_configs():
    """List available environment config files."""
    configs = []
    
    # Check envs/env_configs directory
    env_configs_dir = Path("envs/env_configs")
    if env_configs_dir.exists():
        for f in env_configs_dir.glob("*.json"):
            configs.append(str(f))
    
    # Check converters directory
    converters_dir = Path("converters")
    if converters_dir.exists():
        for f in converters_dir.glob("*.json"):
            configs.append(str(f))
    
    return {"configs": configs}


@app.get("/api/params/wrapper/{name}")
async def get_wrapper_params(name: str):
    """Get parameter info for a wrapper class."""
    if name not in WRAPPER_CLASSES:
        raise HTTPException(status_code=404, detail=f"Unknown wrapper: {name}")
    cls = WRAPPER_CLASSES[name]
    params = _extract_params(cls)
    return {"name": name, "params": params}


@app.get("/api/params/search/{name}")
async def get_search_params(name: str):
    """Get parameter info for a search config class."""
    if name == "none":
        return {"name": name, "params": {}}
    if name not in SEARCH_CLASSES:
        raise HTTPException(status_code=404, detail=f"Unknown search: {name}")
    cls = SEARCH_CLASSES[name]
    params = _extract_params(cls)
    return {"name": name, "params": params}


@app.get("/api/params/agent/{name}")
async def get_agent_params(name: str):
    """Get parameter info for an agent class."""
    if name not in AGENT_CLASSES:
        # Return empty for agents without configurable params
        return {"name": name, "params": {}}
    cls = AGENT_CLASSES[name]
    params = _extract_params(cls)
    return {"name": name, "params": params}


@app.post("/api/session/create")
async def create_session(req: SessionCreateRequest):
    """Create a new session."""
    try:
        logger.debug("create_session request: %s", req)
        session = await manager.create_session(req)
        state = session.get_state()
        return {"session_id": session.sid, "state": state.model_dump()}
    except Exception as e:
        logger.exception("create_session failed")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/session/{sid}/state")
async def get_state(sid: str):
    """Get current session state."""
    try:
        session = await manager.get_session(sid)
        state = session.get_state()
        return {"state": state.model_dump()}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@app.post("/api/session/{sid}/step")
async def step(sid: str, req: StepRequest):
    """Execute a step with the given action."""
    try:
        session = await manager.get_session(sid)
        
        async with session._lock:
            # Save state before step
            session._save_to_history()
            
            # Execute step
            adapter = session.env
            engine = session.pipeline.engine
            adapter.bind(engine)
            obs_dec = adapter.build_observation(session.obs if isinstance(session.obs, dict) else {})
            candidates = adapter.build_candidates(obs_dec)
            mask = candidates.mask.to(dtype=torch.bool, device=adapter.device).view(-1)
            a = int(req.action)

            if int(mask.shape[0]) <= 0 or int(mask.to(torch.int64).sum().item()) == 0:
                session.obs = obs_dec
                reward = float(engine.failure_penalty())
                session.terminated = False
                session.truncated = True
                info = {"reason": "no_valid_actions"}
            elif a < 0 or a >= int(mask.shape[0]):
                session.obs = obs_dec
                reward = float(engine.failure_penalty())
                session.terminated = False
                session.truncated = True
                info = {"reason": "action_out_of_range"}
            elif not bool(mask[a].item()):
                session.obs = obs_dec
                reward = float(engine.failure_penalty())
                session.terminated = False
                session.truncated = True
                info = {"reason": "masked_action"}
            else:
                env_action = adapter.decode_action(a, candidates)
                obs_core, reward, session.terminated, session.truncated, info = engine.step_action(env_action)
                session.obs = adapter.build_observation(obs_core)
            
            # Update candidates for new state
            session._update_candidates()
            
            state = session.get_state()
        
        # Broadcast to WebSocket clients
        await _broadcast(session, {"type": "state", "state": state.model_dump()})
        
        return {"state": state.model_dump(), "reward": float(reward)}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/session/{sid}/undo")
async def undo(sid: str):
    """Undo the last step."""
    try:
        session = await manager.get_session(sid)
        
        async with session._lock:
            if not session.undo():
                raise HTTPException(status_code=400, detail="Nothing to undo")
            state = session.get_state()
        
        await _broadcast(session, {"type": "state", "state": state.model_dump()})
        return {"state": state.model_dump()}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@app.post("/api/session/{sid}/redo")
async def redo(sid: str):
    """Redo the last undone step."""
    try:
        session = await manager.get_session(sid)
        
        async with session._lock:
            if not session.redo():
                raise HTTPException(status_code=400, detail="Nothing to redo")
            state = session.get_state()
        
        await _broadcast(session, {"type": "state", "state": state.model_dump()})
        return {"state": state.model_dump()}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@app.post("/api/session/{sid}/reset")
async def reset(sid: str):
    """Reset the session to initial state."""
    try:
        session = await manager.get_session(sid)
        
        async with session._lock:
            adapter = session.env
            engine = session.pipeline.engine
            adapter.bind(engine)
            _obs_core, _ = engine.reset(options=session.reset_kwargs)
            session.obs = adapter.build_observation(_obs_core)
            session.terminated = False
            session.truncated = False
            session._update_candidates()
            session.history = []
            session.history_index = -1
            session._save_to_history()
            state = session.get_state()
        
        await _broadcast(session, {"type": "state", "state": state.model_dump()})
        return {"state": state.model_dump()}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@app.post("/api/session/{sid}/search")
async def run_search(sid: str, req: SearchRequest):
    """Run MCTS/Beam search with real-time updates."""
    try:
        session = await manager.get_session(sid)
        
        if session.search is None:
            raise HTTPException(status_code=400, detail="No search algorithm configured")
        
        if session.candidates is None:
            raise HTTPException(status_code=400, detail="No candidates available")
        
        # Run search with progress callback
        async def progress_callback(sim: int, visits: np.ndarray, values: np.ndarray, best_action: int):
            """Called during search to broadcast progress."""
            # Build candidate info with visits/values
            candidates = []
            state = session.get_state()
            for i, cand in enumerate(state.candidates):
                cand.visits = int(visits[i]) if i < len(visits) else 0
                cand.q_value = float(values[i]) if i < len(values) else 0.0
                candidates.append(cand)
            
            progress = SearchProgress(
                simulation=sim,
                total=req.simulations,
                candidates=candidates,
                best_action=int(best_action),
                best_value=float(values[best_action]) if best_action < len(values) else 0.0,
            )
            await _broadcast(session, {"type": "search_progress", "progress": progress.model_dump()})
        
        # Run search in background with periodic updates
        result = await _run_search_with_updates(
            session=session,
            simulations=req.simulations,
            broadcast_interval=req.broadcast_interval,
            callback=progress_callback,
        )
        
        return result
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


async def _run_search_with_updates(
    session: Session,
    simulations: int,
    broadcast_interval: int,
    callback,
) -> Dict[str, Any]:
    """Run search with periodic WebSocket updates using unified interface."""
    from search.base import BaseSearch, SearchProgress
    from search.mcts import MCTSSearch, MCTSConfig
    from search.beam import BeamSearch, BeamConfig
    
    search = session.search
    if search is None:
        return {"error": "No search algorithm configured"}
    
    if not isinstance(search, BaseSearch):
        # Fallback for non-BaseSearch implementations:
        # run one decision pass (no env step), then restore state.
        state = {
            "engine": session.pipeline.engine.get_state().copy(),
            "adapter": session.env.get_state_copy(),
        }
        try:
            adapter = session.env
            obs_dec = adapter.build_observation(session.obs if isinstance(session.obs, dict) else {})
            candidates = adapter.build_candidates(obs_dec)
            a = int(session.agent.select_action(obs=obs_dec, candidates=candidates))
            dbg = {
                "action": int(a),
                "search": "fallback_non_base",
                "reason": "fallback_non_base_search",
                "candidates": candidates,
            }
            return {"best_action": int(dbg.get("action", 0)), "debug": dbg}
        finally:
            eng_state = state.get("engine", None)
            adp_state = state.get("adapter", None)
            if isinstance(eng_state, EnvState):
                session.pipeline.engine.set_state(eng_state)
            if isinstance(adp_state, dict):
                session.env.set_state(adp_state)
    
    candidates = session.candidates
    if candidates is None:
        return {"error": "No candidates"}
    
    # Save initial state
    engine = session.pipeline.engine
    adapter = session.env
    initial_state = {"engine": engine.get_state().copy(), "adapter": adapter.get_state_copy()}
    
    # Store latest progress for final response
    latest_progress: Dict[str, Any] = {}
    
    # Configure search with progress callback
    # For MCTS, we need to adjust num_simulations
    if isinstance(search, MCTSSearch):
        # Temporarily override config for this run
        original_config = search.config
        search.config = MCTSConfig(
            num_simulations=simulations,
            c_puct=original_config.c_puct,
            rollout_enabled=original_config.rollout_enabled,
            rollout_depth=original_config.rollout_depth,
            dirichlet_epsilon=original_config.dirichlet_epsilon,
            dirichlet_concentration=original_config.dirichlet_concentration,
            temperature=original_config.temperature,
            pw_enabled=original_config.pw_enabled,
            pw_c=original_config.pw_c,
            pw_alpha=original_config.pw_alpha,
            pw_min_children=original_config.pw_min_children,
            cache_decision_state=original_config.cache_decision_state,
            track_top_k=original_config.track_top_k,
            track_verbose=original_config.track_verbose,
        )
    
    # Set progress callback
    loop = asyncio.get_event_loop()
    
    def threadsafe_callback(progress: SearchProgress) -> None:
        """Thread-safe callback that schedules async broadcast."""
        nonlocal latest_progress
        latest_progress = {
            "iteration": progress.iteration,
            "total": progress.total,
            "visits": progress.visits.tolist(),
            "values": progress.values.tolist(),
            "best_action": progress.best_action,
            "best_value": progress.best_value,
            "extra": progress.extra,
        }
        # Thread-safe way to schedule async task from sync context
        loop.call_soon_threadsafe(
            lambda: asyncio.create_task(_broadcast_progress(session, progress, callback))
        )
    
    search.set_progress_callback(threadsafe_callback, interval=broadcast_interval)
    search.set_adapter(adapter)
    
    def run_search_sync():
        """Run search synchronously (for executor)."""
        return search.select(
            obs=session.obs,
            agent=session.agent,
            root_action_space=candidates,
        )
    
    try:
        # Run search in executor to avoid blocking the event loop
        best_action = await loop.run_in_executor(_search_executor, run_search_sync)
    finally:
        # Clear callback after search
        search.set_progress_callback(None)
        
        # Restore original config for MCTS
        if isinstance(search, MCTSSearch):
            search.config = original_config
    
    # Restore initial state
    eng_state = initial_state.get("engine", None)
    adp_state = initial_state.get("adapter", None)
    if isinstance(eng_state, EnvState):
        engine.set_state(eng_state)
    if isinstance(adp_state, dict):
        adapter.set_state(adp_state)
    
    return {
        "best_action": int(best_action),
        **latest_progress,
    }


async def _broadcast_progress(session: Session, progress, callback) -> None:
    """Broadcast search progress to WebSocket clients."""
    try:
        await callback(
            progress.iteration,
            progress.visits,
            progress.values,
            progress.best_action,
        )
        await asyncio.sleep(0.001)  # Small yield to allow other tasks
    except Exception:
        logger.warning("WebUI Progress broadcast error", exc_info=True)


@app.delete("/api/session/{sid}")
async def delete_session(sid: str):
    """Delete a session."""
    try:
        await manager.delete_session(sid)
        return {"status": "deleted"}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@app.websocket("/ws/{sid}")
async def websocket_endpoint(websocket: WebSocket, sid: str):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    
    try:
        session = await manager.get_session(sid)
        session.websockets.append(websocket)
        
        # Send initial state
        state = session.get_state()
        await websocket.send_json({"type": "state", "state": state.model_dump()})
        
        # Keep connection alive
        while True:
            try:
                data = await websocket.receive_text()
                # Handle ping/pong or other messages
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except WebSocketDisconnect:
                break
    except KeyError:
        await websocket.close(code=4004, reason="Session not found")
    except Exception as e:
        await websocket.close(code=4000, reason=str(e))
    finally:
        try:
            session = await manager.get_session(sid)
            if websocket in session.websockets:
                session.websockets.remove(websocket)
        except:
            pass


async def _broadcast(session: Session, message: Dict[str, Any]) -> None:
    """Broadcast message to all WebSocket clients of a session."""
    dead = []
    for ws in session.websockets:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        try:
            session.websockets.remove(ws)
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
