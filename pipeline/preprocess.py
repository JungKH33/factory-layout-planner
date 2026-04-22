from __future__ import annotations

from dataclasses import dataclass
import importlib
import logging
from pathlib import Path
import shutil
import sys
import time
from typing import Optional

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


def _convert_to_env(input_path: str, output_path: str, visualize_dir: Optional[str]) -> None:
    preprocess_dir = Path(__file__).resolve().parent.parent / "preprocess"
    preprocess_dir_str = str(preprocess_dir)
    if preprocess_dir_str not in sys.path:
        sys.path.insert(0, preprocess_dir_str)
    module = importlib.import_module("to_env")
    convert_fn = getattr(module, "convert_to_env", None)
    if convert_fn is None:
        raise AttributeError("convert_to_env not found in preprocess/to_env.py")
    convert_fn(input_path=input_path, output_path=output_path, visualize_dir=visualize_dir)


def run_preprocess(cfg: PreprocessConfig) -> PreprocessArtifact:
    start = time.perf_counter()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_json_path = _env_output_path(cfg.output_dir)
    _convert_to_env(
        input_path=str(cfg.input_json),
        output_path=str(env_json_path),
        visualize_dir=cfg.visualize_dir,
    )

    input_copy = _input_copy_path(cfg.input_json, cfg.output_dir)
    shutil.copy2(str(cfg.input_json), str(input_copy))

    return PreprocessArtifact(
        stage="preprocess",
        created_at=utc_now_iso(),
        input_json=str(cfg.input_json),
        input_copy_json=str(input_copy),
        env_json=str(env_json_path),
        metrics={
            "elapsed_sec": float(time.perf_counter() - start),
        },
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
