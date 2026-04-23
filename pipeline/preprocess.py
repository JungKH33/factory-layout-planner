from __future__ import annotations

from dataclasses import dataclass
import importlib
import logging
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, Mapping, Optional

from pipeline.schema import PreprocessArtifact, save_json, utc_now_iso

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessConfig:
    input_json: str
    output_dir: str
    visualize_dir: Optional[str] = None


def _env_output_path(output_dir: str) -> Path:
    return Path(output_dir) / "env.json"


def _input_copy_path(input_json: str, output_dir: str) -> Path:
    return Path(output_dir) / Path(input_json).name


def _preprocess_module(name: str):
    preprocess_dir = Path(__file__).resolve().parent.parent / "preprocess"
    preprocess_dir_str = str(preprocess_dir)
    if preprocess_dir_str not in sys.path:
        sys.path.insert(0, preprocess_dir_str)
    return importlib.import_module(name)


def _convert_to_env(input_path: str, output_path: str) -> Dict[str, Any]:
    module = _preprocess_module("to_env")
    convert_fn = getattr(module, "convert_to_env", None)
    if convert_fn is None:
        raise AttributeError("convert_to_env not found in preprocess/to_env.py")
    return convert_fn(input_path=input_path, output_path=output_path)


def _save_variant_visualization(env: Mapping[str, Any], visualize_dir: str) -> Optional[str]:
    module = _preprocess_module("visualize")
    save_fn = getattr(module, "save_variant_images", None)
    if save_fn is None:
        raise AttributeError("save_variant_images not found in preprocess/visualize.py")
    return save_fn(env, visualize_dir)


def run_preprocess(cfg: PreprocessConfig) -> PreprocessArtifact:
    start = time.perf_counter()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_json_path = _env_output_path(cfg.output_dir)
    env_dict = _convert_to_env(
        input_path=str(cfg.input_json),
        output_path=str(env_json_path),
    )

    input_copy = _input_copy_path(cfg.input_json, cfg.output_dir)
    shutil.copy2(str(cfg.input_json), str(input_copy))

    visualization_path: Optional[str] = None
    if cfg.visualize_dir is not None:
        visualization_path = _save_variant_visualization(env_dict, cfg.visualize_dir)
        logger.info("preprocess variant visualization saved: %s", visualization_path)

    metrics: Dict[str, Any] = {"elapsed_sec": float(time.perf_counter() - start)}
    if visualization_path is not None:
        metrics["variant_visualization_path"] = visualization_path

    return PreprocessArtifact(
        stage="preprocess",
        created_at=utc_now_iso(),
        input_json=str(cfg.input_json),
        input_copy_json=str(input_copy),
        env_json=str(env_json_path),
        metrics=metrics,
    )


def run_and_save_preprocess(cfg: PreprocessConfig) -> PreprocessArtifact:
    artifact = run_preprocess(cfg)
    artifact_path = Path(cfg.output_dir) / "preprocess.json"
    save_json(artifact, artifact_path)
    logger.info("preprocess output_dir: %s", cfg.output_dir)
    logger.info("preprocess artifact saved: %s", artifact_path)
    logger.info("preprocess env saved: %s", artifact.env_json)
    logger.info("preprocess input copy saved: %s", artifact.input_copy_json)
    return artifact


__all__ = [
    "PreprocessConfig",
    "run_preprocess",
    "run_and_save_preprocess",
]
