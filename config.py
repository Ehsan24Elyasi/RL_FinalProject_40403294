"""Validated operational configuration for planning commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from agents.q_learning import SHAPING_VERSION, SUPPORTED_SHAPING_METHOD
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
class QLearningRun:
    reward_mode: str
    schedule: str
    seed: int
    episodes: int = 5_000

    def __post_init__(self) -> None:
        if self.reward_mode not in {"sparse", "shaped"}:
            raise ValueError("Q-Learning run reward_mode must be 'sparse' or 'shaped'")
        if self.schedule not in {"linear", "exponential"}:
            raise ValueError("Q-Learning run schedule must be 'linear' or 'exponential'")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Q-Learning run seed must be a nonnegative integer")
        if isinstance(self.episodes, bool) or not isinstance(self.episodes, int) or self.episodes <= 0:
            raise ValueError("Q-Learning run episodes must be a positive integer")

    @property
    def artifact_stem(self) -> str:
        return (
            f"q_{self.reward_mode}_{self.schedule}_seed_{self.seed}"
            f"_ep_{self.episodes}"
        )


@dataclass(frozen=True, slots=True)
class QLearningSettings:
    gamma: float
    alpha: float
    episodes: int
    shaping_method: str
    shaping_version: int
    epsilon_start: float
    epsilon_end: float
    decay_episodes: int
    audit_episode: int
    model_dir: Path
    raw_dir: Path
    required_runs: tuple[QLearningRun, ...]


@dataclass(frozen=True, slots=True)
class SarsaLambdaRun:
    trace_lambda: float
    reward_mode: str
    schedule: str
    seed: int
    episodes: int = 5_000

    def __post_init__(self) -> None:
        if not 0.0 <= self.trace_lambda <= 1.0:
            raise ValueError("SARSA lambda must be in [0, 1]")
        if self.reward_mode not in {"sparse", "shaped"} or self.schedule not in {"linear", "exponential"}:
            raise ValueError("invalid SARSA reward mode or schedule")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("SARSA seed must be a nonnegative integer")
        if isinstance(self.episodes, bool) or not isinstance(self.episodes, int) or self.episodes <= 0:
            raise ValueError("SARSA episodes must be positive")

    @property
    def artifact_stem(self) -> str:
        lam = f"{self.trace_lambda:.1f}".replace(".", "p")
        return f"sarsa_lambda_{lam}_{self.reward_mode}_{self.schedule}_seed_{self.seed}_ep_{self.episodes}"


@dataclass(frozen=True, slots=True)
class SarsaLambdaSettings:
    gamma: float
    alpha: float
    episodes: int
    shaping_method: str
    shaping_version: int
    epsilon_start: float
    epsilon_end: float
    decay_episodes: int
    diagnostic_episode: int
    model_dir: Path
    raw_dir: Path
    required_runs: tuple[SarsaLambdaRun, ...]


@dataclass(frozen=True, slots=True)
class OperationalConfig:
    path: Path
    student_id: str
    base_seed: int
    maze_size: int
    source_map: Path
    rewards: RewardSpec
    planning: PlanningConfig
    q_learning: QLearningSettings
    sarsa_lambda: SarsaLambdaSettings


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not (-float("inf") < result < float("inf")):
        raise ValueError(f"{name} must be finite")
    return result


def _strict_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


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
    q_raw = _mapping(root.get("q_learning"), "q_learning")
    sarsa_raw = _mapping(root.get("sarsa_lambda"), "sarsa_lambda")

    intended = _finite_float(maze.get("intended_probability"), "maze.intended_probability")
    slip = _finite_float(
        maze.get("perpendicular_slip_probability"),
        "maze.perpendicular_slip_probability",
    )
    multiplier = _strict_int(
        maze.get("max_steps_multiplier"), "maze.max_steps_multiplier", minimum=1
    )
    if intended != MazeMDP.INTENDED_PROBABILITY or slip != MazeMDP.SLIP_PROBABILITY:
        raise ValueError(
            "config transition probabilities must match MazeMDP's fixed 0.8/0.1/0.1"
        )
    if multiplier != 3:
        raise ValueError("maze.max_steps_multiplier must match the fixed value 3")

    shaping_method = str(shaping.get("method", ""))
    shaping_version = _strict_int(
        shaping.get("version"), "shaping.version", minimum=1
    )
    if set(shaping) != {"method", "version", "scale"}:
        raise ValueError("shaping must contain exactly method, version, and scale")
    if shaping_method != SUPPORTED_SHAPING_METHOD:
        raise ValueError(
            f"shaping.method must be {SUPPORTED_SHAPING_METHOD!r}"
        )
    if shaping_version != SHAPING_VERSION:
        raise ValueError(f"shaping.version must be {SHAPING_VERSION}")

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
    max_sweeps = _strict_int(
        planning_raw.get("max_sweeps"), "planning.max_sweeps", minimum=1
    )
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

    q_gamma = _finite_float(q_raw.get("gamma"), "q_learning.gamma")
    q_alpha = _finite_float(q_raw.get("alpha"), "q_learning.alpha")
    q_episodes = _strict_int(
        q_raw.get("episodes"), "q_learning.episodes", minimum=1
    )
    epsilon_start = _finite_float(
        q_raw.get("epsilon_start"), "q_learning.epsilon_start"
    )
    epsilon_end = _finite_float(q_raw.get("epsilon_end"), "q_learning.epsilon_end")
    decay_episodes = _strict_int(
        q_raw.get("decay_episodes"), "q_learning.decay_episodes", minimum=2
    )
    audit_episode = _strict_int(
        q_raw.get("audit_episode"), "q_learning.audit_episode", minimum=1
    )
    if not 0.0 <= q_gamma < 1.0:
        raise ValueError("q_learning.gamma must be in [0, 1)")
    if not 0.0 < q_alpha <= 1.0:
        raise ValueError("q_learning.alpha must be in (0, 1]")
    if q_episodes <= 0:
        raise ValueError("q_learning.episodes must be positive")
    if not 0.0 <= epsilon_end <= epsilon_start <= 1.0:
        raise ValueError(
            "q_learning epsilon values must satisfy 0 <= end <= start <= 1"
        )
    if decay_episodes < 2:
        raise ValueError("q_learning.decay_episodes must be at least 2")
    if not 1 <= audit_episode <= q_episodes:
        raise ValueError("q_learning.audit_episode must be within episodes")

    q_runs_raw = q_raw.get("required_runs")
    if not isinstance(q_runs_raw, list) or not q_runs_raw:
        raise ValueError("q_learning.required_runs must be a nonempty list")
    def parse_q_run(item: Any) -> QLearningRun:
        run = _mapping(item, "Q-Learning run")
        return QLearningRun(
            reward_mode=str(run.get("reward_mode")),
            schedule=str(run.get("schedule")),
            seed=_strict_int(run.get("seed"), "Q-Learning run seed", minimum=0),
            episodes=_strict_int(
                run.get("episodes", q_episodes),
                "Q-Learning run episodes",
                minimum=1,
            ),
        )

    q_runs = tuple(parse_q_run(item) for item in q_runs_raw)
    expected_q_runs = (
        QLearningRun("shaped", "linear", 9, 5_000),
        QLearningRun("shaped", "exponential", 9, 5_000),
    )
    if q_runs != expected_q_runs:
        raise ValueError(
            "q_learning.required_runs must be shaped linear/exponential, seed 9, 5000 episodes"
        )

    s_gamma = _finite_float(sarsa_raw.get("gamma"), "sarsa_lambda.gamma")
    s_alpha = _finite_float(sarsa_raw.get("alpha"), "sarsa_lambda.alpha")
    s_episodes = _strict_int(sarsa_raw.get("episodes"), "sarsa_lambda.episodes", minimum=1)
    s_decay = _strict_int(sarsa_raw.get("decay_episodes"), "sarsa_lambda.decay_episodes", minimum=2)
    s_diag = _strict_int(sarsa_raw.get("diagnostic_episode"), "sarsa_lambda.diagnostic_episode", minimum=1)
    s_eps_start = _finite_float(sarsa_raw.get("epsilon_start"), "sarsa_lambda.epsilon_start")
    s_eps_end = _finite_float(sarsa_raw.get("epsilon_end"), "sarsa_lambda.epsilon_end")
    if not 0 <= s_gamma < 1 or not 0 < s_alpha <= 1 or not 0 <= s_eps_end <= s_eps_start <= 1 or not 1 <= s_diag <= s_episodes:
        raise ValueError("invalid sarsa_lambda primary settings")
    s_runs_raw = sarsa_raw.get("required_runs")
    if not isinstance(s_runs_raw, list):
        raise ValueError("sarsa_lambda.required_runs must be a list")
    s_runs = tuple(SarsaLambdaRun(
        _finite_float(_mapping(item, "SARSA run").get("lambda"), "SARSA run lambda"),
        str(_mapping(item, "SARSA run").get("reward_mode")),
        str(_mapping(item, "SARSA run").get("schedule")),
        _strict_int(_mapping(item, "SARSA run").get("seed"), "SARSA run seed", minimum=0),
        _strict_int(_mapping(item, "SARSA run").get("episodes", s_episodes), "SARSA run episodes", minimum=1),
    ) for item in s_runs_raw)
    expected_s = tuple(SarsaLambdaRun(value, "shaped", "linear", 9, 5000) for value in (0.0, 0.3, 0.7, 0.9))
    if s_runs != expected_s:
        raise ValueError("sarsa_lambda.required_runs must use lambdas 0, .3, .7, .9 in order")

    base = config_path.parent
    source_map = Path(str(maze.get("source_map")))
    output_dir = Path(str(planning_raw.get("output_dir")))
    q_model_dir = Path(str(q_raw.get("model_dir")))
    q_raw_dir = Path(str(q_raw.get("raw_dir")))
    s_model_dir = Path(str(sarsa_raw.get("model_dir")))
    s_raw_dir = Path(str(sarsa_raw.get("raw_dir")))
    if not source_map.is_absolute():
        source_map = (base / source_map).resolve()
    if not output_dir.is_absolute():
        output_dir = (base / output_dir).resolve()
    if not q_model_dir.is_absolute():
        q_model_dir = (base / q_model_dir).resolve()
    if not q_raw_dir.is_absolute():
        q_raw_dir = (base / q_raw_dir).resolve()
    if not s_model_dir.is_absolute():
        s_model_dir = (base / s_model_dir).resolve()
    if not s_raw_dir.is_absolute():
        s_raw_dir = (base / s_raw_dir).resolve()

    student_id = str(project.get("student_id", "")).strip()
    if not student_id:
        raise ValueError("project.student_id must not be empty")
    base_seed = _strict_int(project.get("base_seed"), "project.base_seed", minimum=0)
    maze_size = _strict_int(maze.get("size"), "maze.size", minimum=1)
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
        q_learning=QLearningSettings(
            gamma=q_gamma,
            alpha=q_alpha,
            episodes=q_episodes,
            shaping_method=shaping_method,
            shaping_version=shaping_version,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            decay_episodes=decay_episodes,
            audit_episode=audit_episode,
            model_dir=q_model_dir,
            raw_dir=q_raw_dir,
            required_runs=q_runs,
        ),
        sarsa_lambda=SarsaLambdaSettings(
            gamma=s_gamma, alpha=s_alpha, episodes=s_episodes,
            shaping_method=shaping_method, shaping_version=shaping_version,
            epsilon_start=s_eps_start, epsilon_end=s_eps_end,
            decay_episodes=s_decay, diagnostic_episode=s_diag,
            model_dir=s_model_dir, raw_dir=s_raw_dir, required_runs=s_runs,
        ),
    )
