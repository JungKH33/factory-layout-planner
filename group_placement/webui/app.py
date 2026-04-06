"""FastAPI WebUI application for Factory Layout."""
from __future__ import annotations

import asyncio
import dataclasses
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

from group_placement.envs.action_space import ActionSpace
from group_placement.agents.placement.greedy import GreedyAgent, GreedyAdapter, GreedyV2Adapter, GreedyV3Adapter
from group_placement.agents.placement.alphachip import AlphaChipAdapter
from group_placement.agents.placement.maskplace import MaskPlaceAdapter
from group_placement.search.mcts import MCTSConfig, MCTSSearch
from group_placement.search.beam import BeamConfig, BeamSearch


def _extract_params(cls, exclude: set = None) -> Dict[str, Dict[str, Any]]:
    """Extract parameter info from class __init__ signature or dataclass fields."""
    import dataclasses as dc
    exclude = exclude or set()
    params = {}

    # Check if it's a dataclass
    if dc.is_dataclass(cls):
        for field in dc.fields(cls):
            name = field.name
            if name in exclude or name.startswith('_'):
                continue

            info = {"name": name}

            if field.default is not dc.MISSING:
                info["default"] = field.default
                info["required"] = False
            elif field.default_factory is not dc.MISSING:
                info["default"] = None
                info["required"] = False
            else:
                info["default"] = None
                info["required"] = True

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

        if param.default is not inspect.Parameter.empty:
            info["default"] = param.default
            info["required"] = False
        else:
            info["default"] = None
            info["required"] = True

        if param.annotation is not inspect.Parameter.empty:
            ann = param.annotation
            if hasattr(ann, '__origin__'):
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

from group_placement.webui.schemas import (
    SessionCreateRequest,
    StepRequest,
    SearchRequest,
    GotoRequest,
    TopKRestoreRequest,
    SessionState,
    CandidateInfo,
    SearchProgress,
)
from group_placement.webui.session import manager, Session


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

    env_configs_dir = Path("envs/env_configs")
    if env_configs_dir.exists():
        for f in env_configs_dir.glob("*.json"):
            configs.append(str(f))

    converters_dir = Path("converters")
    if converters_dir.exists():
        for f in converters_dir.glob("*.json"):
            configs.append(str(f))

    return {"configs": configs}


@app.get("/api/params/wrapper/{name}")
async def get_wrapper_params(name: str):
    if name not in WRAPPER_CLASSES:
        raise HTTPException(status_code=404, detail=f"Unknown wrapper: {name}")
    cls = WRAPPER_CLASSES[name]
    params = _extract_params(cls)
    return {"name": name, "params": params}


@app.get("/api/params/search/{name}")
async def get_search_params(name: str):
    if name == "none":
        return {"name": name, "params": {}}
    if name not in SEARCH_CLASSES:
        raise HTTPException(status_code=404, detail=f"Unknown search: {name}")
    cls = SEARCH_CLASSES[name]
    params = _extract_params(cls)
    return {"name": name, "params": params}


@app.get("/api/params/agent/{name}")
async def get_agent_params(name: str):
    if name not in AGENT_CLASSES:
        return {"name": name, "params": {}}
    cls = AGENT_CLASSES[name]
    params = _extract_params(cls)
    return {"name": name, "params": params}


@app.post("/api/session/create")
async def create_session(req: SessionCreateRequest):
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
            if session.search_in_progress:
                raise HTTPException(status_code=409, detail="Search is running; try again after it completes")
            explorer = session.explorer
            node = explorer.current()

            if node.terminal:
                raise HTTPException(status_code=400, detail="Episode already terminated")

            try:
                child = explorer.step(req.action, chosen_by="human")
            except (IndexError, ValueError) as e:
                raise HTTPException(status_code=400, detail=str(e))

            reward = child.cum_reward - node.cum_reward

            # Predict agent for the new state (populates candidates/scores)
            if not child.terminal:
                explorer.predict_agent()

            state = session.get_state()

        await _broadcast(session, {"type": "state", "state": state.model_dump()})
        return {"state": state.model_dump(), "reward": float(reward)}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/session/{sid}/undo")
async def undo(sid: str):
    try:
        session = await manager.get_session(sid)

        async with session._lock:
            if session.search_in_progress:
                raise HTTPException(status_code=409, detail="Search is running; try again after it completes")
            if not session.undo():
                raise HTTPException(status_code=400, detail="Nothing to undo")
            # Re-predict agent for restored state
            if not session.explorer.current().terminal:
                session.explorer.predict_agent()
            state = session.get_state()

        await _broadcast(session, {"type": "state", "state": state.model_dump()})
        return {"state": state.model_dump()}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@app.post("/api/session/{sid}/redo")
async def redo(sid: str):
    try:
        session = await manager.get_session(sid)

        async with session._lock:
            if session.search_in_progress:
                raise HTTPException(status_code=409, detail="Search is running; try again after it completes")
            if not session.redo():
                raise HTTPException(status_code=400, detail="Nothing to redo")
            if not session.explorer.current().terminal:
                session.explorer.predict_agent()
            state = session.get_state()

        await _broadcast(session, {"type": "state", "state": state.model_dump()})
        return {"state": state.model_dump()}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@app.post("/api/session/{sid}/reset")
async def reset(sid: str):
    try:
        session = await manager.get_session(sid)

        async with session._lock:
            if session.search_in_progress:
                raise HTTPException(status_code=409, detail="Search is running; try again after it completes")
            session.explorer.reset(options=session.reset_kwargs)
            session.search_timeline.clear()
            session.topk_runs.clear()
            session.topk_state_map.clear()
            session.topk_run_seq = 0
            if not session.explorer.current().terminal:
                session.explorer.predict_agent()
            state = session.get_state()

        await _broadcast(session, {"type": "state", "state": state.model_dump()})
        return {"state": state.model_dump()}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@app.get("/api/session/{sid}/tree")
async def get_tree(sid: str):
    try:
        session = await manager.get_session(sid)
        async with session._lock:
            tree = session.explorer.export_tree()
        return {"tree": tree}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@app.get("/api/session/{sid}/signals")
async def get_signals(sid: str):
    try:
        session = await manager.get_session(sid)
        async with session._lock:
            payload = _signals_payload(session)
        return payload
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@app.get("/api/session/{sid}/search_timeline")
async def get_search_timeline(sid: str):
    try:
        session = await manager.get_session(sid)
        async with session._lock:
            return {"timeline": list(session.search_timeline)}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@app.get("/api/session/{sid}/topk")
async def get_topk(sid: str):
    try:
        session = await manager.get_session(sid)
        async with session._lock:
            return {"runs": list(session.topk_runs)}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@app.post("/api/session/{sid}/topk/restore")
async def restore_topk(sid: str, req: TopKRestoreRequest):
    try:
        session = await manager.get_session(sid)
        async with session._lock:
            if session.search_in_progress:
                raise HTTPException(status_code=409, detail="Search is running; try again after it completes")
            item = session.topk_state_map.get(req.item_id)
            if item is None:
                raise HTTPException(status_code=404, detail=f"Top-K item not found: {req.item_id}")
            engine_state = item.get("engine_state")
            if engine_state is None:
                raise HTTPException(status_code=400, detail="Top-K item does not contain restorable engine state")

            exp = session.explorer
            exp.engine.set_state(engine_state)
            exp.adapter.set_state(exp.adapter.get_state_copy())
            root = exp.rebase_to_current_state()
            if not root.terminal:
                exp.predict_agent()

            state = session.get_state()
            tree = exp.export_tree()

        await _broadcast(session, {"type": "state", "state": state.model_dump()})
        return {"state": state.model_dump(), "tree": tree}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@app.post("/api/session/{sid}/goto")
async def goto_node(sid: str, req: GotoRequest):
    try:
        session = await manager.get_session(sid)
        async with session._lock:
            if session.search_in_progress:
                raise HTTPException(status_code=409, detail="Search is running; try again after it completes")
            try:
                node = session.explorer.goto(int(req.node_id))
            except KeyError:
                raise HTTPException(status_code=404, detail=f"Node not found: {req.node_id}")

            if (not node.terminal) and ("agent" not in node.signals):
                session.explorer.predict_agent()

            state = session.get_state()
            tree = session.explorer.export_tree()

        await _broadcast(session, {"type": "state", "state": state.model_dump()})
        return {"state": state.model_dump(), "tree": tree}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@app.post("/api/session/{sid}/search")
async def run_search(sid: str, req: SearchRequest):
    """Run search with real-time WebSocket updates."""
    try:
        session = await manager.get_session(sid)
        async with session._lock:
            explorer = session.explorer
            if session.search_in_progress:
                raise HTTPException(status_code=409, detail="Search is already running")
            if explorer.search is None:
                raise HTTPException(status_code=400, detail="No search algorithm configured")
            if explorer.current().terminal:
                raise HTTPException(status_code=400, detail="Episode already terminated")
            session.search_in_progress = True

        try:
            result = await _run_search_with_updates(
                session=session,
                simulations=req.simulations,
                broadcast_interval=req.broadcast_interval,
            )
            return result
        finally:
            async with session._lock:
                session.search_in_progress = False
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


async def _run_search_with_updates(
    session: Session,
    simulations: int,
    broadcast_interval: int,
) -> Dict[str, Any]:
    """Run search via Explorer with periodic WebSocket updates."""
    explorer = session.explorer
    search = explorer.search
    if search is None:
        return {"error": "No search algorithm configured"}

    # Temporarily override simulation count for MCTS
    original_config = None
    if isinstance(search, MCTSSearch):
        original_config = search.config
        search.config = dataclasses.replace(original_config, num_simulations=simulations)

    loop = asyncio.get_event_loop()
    latest_progress: Dict[str, Any] = {}
    progress_candidates = _build_progress_candidates(session)
    timeline_local: List[Dict[str, Any]] = []

    # Register event listener that bridges sync search → async WebSocket
    from group_placement.trace.schema import TraceEvent

    def on_event(event: TraceEvent) -> None:
        if event.type != "search_progress":
            return
        nonlocal latest_progress
        latest_progress = event.data

        # Build candidate progress for WebSocket
        data = event.data
        visits = data.get("visits", [])
        values = data.get("values", [])
        best_action = data.get("best_action", 0)
        best_value = data.get("best_value", 0.0)
        timeline_local.append({
            "iteration": int(data.get("iteration", 0)),
            "total": int(data.get("total", simulations)),
            "best_action": int(best_action),
            "best_value": float(best_value),
        })

        def _schedule_broadcast():
            asyncio.create_task(_broadcast_search_progress(
                session, data.get("iteration", 0), simulations,
                visits, values, best_action, best_value, progress_candidates,
            ))

        loop.call_soon_threadsafe(_schedule_broadcast)

    explorer.on(on_event)

    def run_sync():
        return explorer.predict_search(progress_interval=broadcast_interval)

    try:
        signal = await loop.run_in_executor(_search_executor, run_sync)
    finally:
        explorer.off(on_event)
        if original_config is not None:
            search.config = original_config

    async with session._lock:
        session.search_timeline = timeline_local[-2000:]
        _store_topk_results(session, signal)

    return {
        "best_action": signal.recommended_action,
        "top_k_count": len(signal.metadata.get("top_k", []) if isinstance(signal.metadata.get("top_k"), list) else []),
        **latest_progress,
    }


async def _broadcast_search_progress(
    session: Session,
    iteration: int,
    total: int,
    visits: list,
    values: list,
    best_action: int,
    best_value: float,
    base_candidates: List[CandidateInfo],
) -> None:
    """Broadcast search progress to WebSocket clients."""
    try:
        # Use immutable candidate snapshot captured at search start.
        candidates = [
            CandidateInfo(**cand.model_dump())
            for cand in base_candidates
        ]
        for i, cand in enumerate(candidates):
            cand.visits = int(visits[i]) if i < len(visits) else 0
            cand.q_value = float(values[i]) if i < len(values) else 0.0

        progress = SearchProgress(
            simulation=iteration,
            total=total,
            candidates=candidates,
            best_action=int(best_action),
            best_value=float(best_value),
        )
        await _broadcast(session, {"type": "search_progress", "progress": progress.model_dump()})
        await asyncio.sleep(0.001)
    except Exception:
        logger.warning("WebUI Progress broadcast error", exc_info=True)


def _build_progress_candidates(session: Session) -> List[CandidateInfo]:
    node = session.explorer.current()
    if node.terminal:
        return []
    action_space = node._snapshot.action_space if (node._snapshot and node._snapshot.action_space is not None) else None
    if action_space is None:
        return []
    agent_sig = node.signals.get("agent")
    scores = agent_sig.scores if agent_sig is not None else None
    mask = action_space.valid_mask.detach().cpu().numpy()
    poses = action_space.centers.detach().cpu().numpy()
    out: List[CandidateInfo] = []
    for i in range(len(poses)):
        out.append(CandidateInfo(
            index=i,
            x=float(poses[i, 0]),
            y=float(poses[i, 1]),
            rot=0,
            score=float(scores[i]) if (scores is not None and i < len(scores)) else 0.0,
            valid=bool(mask[i]),
            visits=0,
            q_value=0.0,
        ))
    return out


def _signals_payload(session: Session) -> Dict[str, Any]:
    def _safe_value(v: Any) -> Any:
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (int, float, str, bool, type(None))):
            return v
        if isinstance(v, list):
            return [_safe_value(x) for x in v]
        if isinstance(v, dict):
            return {str(k): _safe_value(x) for k, x in v.items()}
        return str(v)

    node = session.explorer.current()
    signals_out: Dict[str, Any] = {}
    for src, sig in node.signals.items():
        scores = np.array(sig.scores, dtype=np.float32) if sig.scores is not None else np.zeros((0,), dtype=np.float32)
        top_idx = np.argsort(-scores)[:5].tolist() if scores.size > 0 else []
        top_actions = [
            {"action": int(i), "score": float(scores[i])}
            for i in top_idx if 0 <= int(i) < int(scores.size)
        ]
        signals_out[src] = {
            "recommended_action": int(sig.recommended_action),
            "recommended_value": float(sig.recommended_value),
            "top_actions": top_actions,
            "metadata": _safe_value(sig.metadata),
        }

    agent = signals_out.get("agent")
    search = None
    for k in signals_out.keys():
        if str(k).startswith("search:"):
            search = signals_out[k]
            break
    agree = None
    if agent is not None and search is not None:
        a = int(agent["recommended_action"])
        b = int(search["recommended_action"])
        if a >= 0 and b >= 0:
            agree = bool(a == b)

    return {"signals": signals_out, "agreement": agree}


def _store_topk_results(session: Session, signal: Any) -> None:
    top_k = signal.metadata.get("top_k", None) if hasattr(signal, "metadata") else None
    if not isinstance(top_k, list) or len(top_k) == 0:
        return

    session.topk_run_seq += 1
    run_id = int(session.topk_run_seq)
    run_items: List[Dict[str, Any]] = []
    for rank, item in enumerate(top_k, start=1):
        if not isinstance(item, dict):
            continue
        item_id = f"run{run_id}_k{rank}"
        session.topk_state_map[item_id] = item
        run_items.append({
            "item_id": item_id,
            "rank": int(rank),
            "cost": float(item.get("cost", 0.0)),
            "cum_reward": float(item.get("cum_reward", 0.0)),
            "placed": int(len(item.get("positions", {}))) if isinstance(item.get("positions"), dict) else 0,
        })

    session.topk_runs.insert(0, {
        "run_id": run_id,
        "source": str(getattr(signal, "source", "")),
        "items": run_items,
    })
    session.topk_runs = session.topk_runs[:20]


@app.delete("/api/session/{sid}")
async def delete_session(sid: str):
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
