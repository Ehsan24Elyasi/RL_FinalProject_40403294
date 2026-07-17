"""Validated operational configuration for planning commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from environments.maze import MazeMDP, RewardSpec

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


@dataclass(frozen=True, slots=True)
class PlanningRun:
    reward_mode: str
    gamma: float

    def __post_init__(self) -> None:
        if self.reward_mode not in {"sparse", "shaped"}:
            raise ValueError("planning run reward_mode must be 'sparse' or 'shaped'")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("planning run gamma must be in [0, 1)")

    @property
    def filename(self) -> str:
        gamma_text = f"{self.gamma:.2f}".replace(".", "p")
        return f"vi_{self.reward_mode}_gamma_{gamma_text}.npz"


@dataclass(frozen=True, slots=True)
class PlanningConfig:
    theta: float
    max_sweeps: int
    tie_tolerance: float
    output_dir: Path
    required_runs: tuple[PlanningRun, ...]


@dataclass(frozen=True, slots=True)
class OperationalConfig:
    path: Path
    student_id: str
    base_seed: int
    maze_size: int
    source_map: Path
    rewards: RewardSpec
    planning: PlanningConfig


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not (-float("inf") < result < float("inf")):
        raise ValueError(f"{name} must be finite")
    return result


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> OperationalConfig:
    """Load YAML, validate fixed environment facts, and resolve relative paths."""

    config_path = Path(path).resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read config {config_path}: {exc}") from exc
    root = _mapping(raw, "config root")
    project = _mapping(root.get("project"), "project")
    maze = _mapping(root.get("maze"), "maze")
    rewards_raw = _mapping(root.get("rewards"), "rewards")
    shaping = _mapping(root.get("shaping"), "shaping")
    planning_raw = _mapping(root.get("planning"), "planning")

    intended = _finite_float(maze.get("intended_probability"), "maze.intended_probability")
    slip = _finite_float(
        maze.get("perpendicular_slip_probability"),
        "maze.perpendicular_slip_probability",
    )
    multiplier = int(maze.get("max_steps_multiplier", 0))
    if intended != MazeMDP.INTENDED_PROBABILITY or slip != MazeMDP.SLIP_PROBABILITY:
        raise ValueError(
            "config transition probabilities must match MazeMDP's fixed 0.8/0.1/0.1"
        )
    if multiplier != 3:
        raise ValueError("maze.max_steps_multiplier must match the fixed value 3")

    reward_spec = RewardSpec(
        step=_finite_float(rewards_raw.get("step"), "rewards.step"),
        collision=_finite_float(
            rewards_raw.get("collision_extra"), "rewards.collision_extra"
        ),
        penalty=_finite_float(
            rewards_raw.get("penalty_extra"), "rewards.penalty_extra"
        ),
        key=_finite_float(rewards_raw.get("first_key"), "rewards.first_key"),
        goal=_finite_float(rewards_raw.get("goal"), "rewards.goal"),
        shaping_scale=_finite_float(shaping.get("scale"), "shaping.scale"),
    )

    theta = _finite_float(planning_raw.get("theta"), "planning.theta")
    tie_tolerance = _finite_float(
        planning_raw.get("tie_tolerance"), "planning.tie_tolerance"
    )
    max_sweeps = int(planning_raw.get("max_sweeps", 0))
    if theta <= 0.0:
        raise ValueError("planning.theta must be positive")
    if tie_tolerance < 0.0:
        raise ValueError("planning.tie_tolerance must be nonnegative")
    if max_sweeps <= 0:
        raise ValueError("planning.max_sweeps must be positive")

    runs_raw = planning_raw.get("required_runs")
    if not isinstance(runs_raw, list) or not runs_raw:
        raise ValueError("planning.required_runs must be a nonempty list")
    runs = tuple(
        PlanningRun(
            reward_mode=str(_mapping(item, "planning run").get("reward_mode")),
            gamma=_finite_float(
                _mapping(item, "planning run").get("gamma"), "planning run gamma"
            ),
        )
        for item in runs_raw
    )
    expected = (
        PlanningRun("shaped", 0.90),
        PlanningRun("shaped", 0.95),
        PlanningRun("shaped", 0.99),
        PlanningRun("sparse", 0.95),
    )
    if runs != expected:
        raise ValueError("planning.required_runs must be shaped .90/.95/.99 then sparse .95")

    base = config_path.parent
    source_map = Path(str(maze.get("source_map")))
    output_dir = Path(str(planning_raw.get("output_dir")))
    if not source_map.is_absolute():
        source_map = (base / source_map).resolve()
    if not output_dir.is_absolute():
        output_dir = (base / output_dir).resolve()

    student_id = str(project.get("student_id", "")).strip()
    if not student_id:
        raise ValueError("project.student_id must not be empty")
    base_seed = int(project.get("base_seed"))
    maze_size = int(maze.get("size"))
    if maze_size <= 0:
        raise ValueError("maze.size must be positive")

    return OperationalConfig(
        path=config_path,
        student_id=student_id,
        base_seed=base_seed,
        maze_size=maze_size,
        source_map=source_map,
        rewards=reward_spec,
        planning=PlanningConfig(
            theta=theta,
            max_sweeps=max_sweeps,
            tie_tolerance=tie_tolerance,
            output_dir=output_dir,
            required_runs=runs,
        ),
    )
