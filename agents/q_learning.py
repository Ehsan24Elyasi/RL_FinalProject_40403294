"""Manual off-policy Q-Learning with reproducible, auditable artifact bundles."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping
import uuid

import numpy as np

from environments.maze import Action, EventType, MazeEpisode, MazeMDP, MazeSpec, State
from .common import (
    ACTION_NAMES,
    ACTION_ORDER,
    LoadedQLearning,
    dense_q_array,
    load_q_learning_npz,
    map_checksum,
    reachable_state_mask,
    save_q_learning_npz,
    state_index,
    terminal_state_mask,
    valid_state_mask,
)

SCHEDULES = frozenset({"linear", "exponential", "geometric"})
REWARD_MODES = frozenset({"sparse", "shaped"})
SUPPORTED_SHAPING_METHOD = "normalized_completion_distance"
SHAPING_VERSION = 1
RUN_CONFIG_SCHEMA_VERSION = 1
CSV_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
ALGORITHM_ID = "manual_off_policy_q_learning"
SEED_DERIVATION = "numpy.SeedSequence(root).spawn(2); uint64 child state"
BEHAVIOR_POLICY = "epsilon_greedy_uniform_exploration_uniform_exact_max_ties"
Q_INITIALIZATION = "valid_zero_walls_nan_terminal_zero"
ACTION_INDEX = {action: index for index, action in enumerate(ACTION_ORDER)}


@dataclass(frozen=True, slots=True)
class QLearningConfig:
    gamma: float = 0.95
    alpha: float = 0.1
    episodes: int = 5_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    decay_episodes: int = 4_000
    schedule: str = "linear"
    reward_mode: str = "shaped"
    audit_episode: int = 1
    shaping_method: str = SUPPORTED_SHAPING_METHOD
    shaping_version: int = SHAPING_VERSION

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must be in [0, 1)")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if isinstance(self.episodes, bool) or self.episodes <= 0:
            raise ValueError("episodes must be a positive integer")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
        if isinstance(self.decay_episodes, bool) or self.decay_episodes < 2:
            raise ValueError("decay_episodes must be an integer of at least 2")
        if self.schedule not in SCHEDULES:
            raise ValueError("schedule must be 'linear', 'exponential', or 'geometric'")
        if self.reward_mode not in REWARD_MODES:
            raise ValueError("reward_mode must be 'sparse' or 'shaped'")
        if isinstance(self.audit_episode, bool) or not 1 <= self.audit_episode <= self.episodes:
            raise ValueError("audit_episode must be within the configured episodes")
        if self.schedule in {"exponential", "geometric"} and self.epsilon_end == 0.0:
            raise ValueError("exponential/geometric epsilon_end must be positive")
        if self.shaping_method != SUPPORTED_SHAPING_METHOD:
            raise ValueError(f"shaping_method must be {SUPPORTED_SHAPING_METHOD!r}")
        if self.shaping_version != SHAPING_VERSION:
            raise ValueError(f"shaping_version must be {SHAPING_VERSION}")


@dataclass(frozen=True, slots=True)
class QLearningSeeds:
    root: int
    behavior: int
    transition: int

    def __post_init__(self) -> None:
        values = (self.root, self.behavior, self.transition)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            raise ValueError("Q-Learning seeds must be nonnegative integers")
        if self.behavior == self.transition:
            raise ValueError("behavior and transition seeds must be independent")


@dataclass(frozen=True, slots=True)
class QLearningRunIdentity:
    config_json: str
    semantic_config_hash: str
    run_id: str

    def __post_init__(self) -> None:
        try:
            parsed = json.loads(self.config_json)
        except json.JSONDecodeError as exc:
            raise ValueError("run configuration JSON is invalid") from exc
        if _canonical_json(parsed) != self.config_json:
            raise ValueError("run configuration JSON must be canonical")
        digest = hashlib.sha256(self.config_json.encode("utf-8")).hexdigest()
        if digest != self.semantic_config_hash:
            raise ValueError("semantic config hash does not match config JSON")
        if self.run_id != f"q-learning-{digest}":
            raise ValueError("run_id does not match semantic config hash")

    @property
    def short_id(self) -> str:
        return self.semantic_config_hash[:12]


@dataclass(frozen=True, slots=True)
class ActionSelection:
    action: Action
    action_index: int
    epsilon: float
    exploring: bool
    greedy_action_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not 0 <= self.action_index < len(ACTION_ORDER):
            raise ValueError("action_index is out of range")
        if ACTION_ORDER[self.action_index] is not self.action:
            raise ValueError("action and action_index disagree")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if not self.greedy_action_indices:
            raise ValueError("at least one greedy action is required")
        if any(index < 0 or index >= len(ACTION_ORDER) for index in self.greedy_action_indices):
            raise ValueError("greedy action index is out of range")
        if not self.exploring and self.action_index not in self.greedy_action_indices:
            raise ValueError("a greedy selection must choose an exact-max action")


@dataclass(frozen=True, slots=True)
class QUpdate:
    state: State
    intended_action: Action
    action_index: int
    reward: float
    next_state: State
    old_q: float
    next_max_q: float
    bootstrap_value: float
    target: float
    td_error: float
    alpha: float
    gamma: float
    new_q: float
    terminated: bool
    truncated: bool

    def __post_init__(self) -> None:
        numeric = (self.reward, self.old_q, self.next_max_q, self.bootstrap_value,
                   self.target, self.td_error, self.alpha, self.gamma, self.new_q)
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("Q update values must be finite")
        if self.terminated and self.truncated:
            raise ValueError("a transition cannot terminate and truncate")
        if not 0 <= self.action_index < len(ACTION_ORDER):
            raise ValueError("action_index is out of range")
        if ACTION_ORDER[self.action_index] is not self.intended_action:
            raise ValueError("intended action and action index disagree")
        if self.terminated and self.bootstrap_value != 0.0:
            raise ValueError("terminal Q updates must not bootstrap")
        if not 0.0 < self.alpha <= 1.0 or not 0.0 <= self.gamma < 1.0:
            raise ValueError("invalid alpha or gamma in Q update")


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    episode: int
    epsilon: float
    steps: int
    base_return: float
    shaping_return: float
    total_return: float
    learning_return: float
    success: bool
    terminated: bool
    truncated: bool
    move_count: int
    wall_collision_count: int
    penalty_entered_count: int
    key_collected_count: int
    closed_door_attempt_count: int
    door_passed_count: int
    teleported_count: int
    goal_reached_count: int
    episode_truncated_count: int
    unique_state_visits: int
    repeated_state_visits: int
    runtime_seconds: float

    def __post_init__(self) -> None:
        if self.episode <= 0 or self.steps <= 0:
            raise ValueError("episode and steps must be positive")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if self.terminated and self.truncated:
            raise ValueError("an episode cannot terminate and truncate")
        if not (self.terminated or self.truncated):
            raise ValueError("a completed episode must terminate or truncate")
        if self.success != self.terminated:
            raise ValueError("success must match goal termination")
        counts = (self.move_count, self.wall_collision_count, self.penalty_entered_count,
                  self.key_collected_count, self.closed_door_attempt_count,
                  self.door_passed_count, self.teleported_count,
                  self.goal_reached_count, self.episode_truncated_count)
        if any(count < 0 for count in counts):
            raise ValueError("event counts must be nonnegative")
        if self.unique_state_visits <= 0 or self.repeated_state_visits < 0:
            raise ValueError("state-visit metrics are invalid")
        if self.unique_state_visits + self.repeated_state_visits != self.steps + 1:
            raise ValueError("state-visit metrics must include initial and resulting states")
        if not math.isclose(self.total_return, self.base_return + self.shaping_return,
                            rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("total return must equal base plus shaping return")
        if not math.isfinite(self.runtime_seconds) or self.runtime_seconds < 0.0:
            raise ValueError("episode runtime must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class AuditRow:
    episode: int
    step: int
    epsilon: float
    exploring: bool
    source_key: int
    source_row: int
    source_col: int
    intended_action: str
    intended_action_index: int
    actual_action: str
    actual_action_index: int
    transition_probability: float
    next_key: int
    next_row: int
    next_col: int
    events: str
    base_reward: float
    shaping_reward: float
    total_reward: float
    learning_reward: float
    old_q: float
    next_max_q: float
    bootstrap_value: float
    gamma: float
    target: float
    td_error: float
    alpha: float
    new_q: float
    terminated: bool
    truncated: bool

    def __post_init__(self) -> None:
        if self.episode <= 0 or self.step <= 0:
            raise ValueError("audit episode and step must be positive")
        if self.source_key not in (0, 1) or self.next_key not in (0, 1):
            raise ValueError("audit key indices must be 0 or 1")
        if not 0 <= self.intended_action_index < len(ACTION_ORDER):
            raise ValueError("audit intended action index is out of range")
        if not 0 <= self.actual_action_index < len(ACTION_ORDER):
            raise ValueError("audit actual action index is out of range")
        if self.intended_action != ACTION_ORDER[self.intended_action_index].name:
            raise ValueError("audit intended action metadata disagrees")
        if self.actual_action != ACTION_ORDER[self.actual_action_index].name:
            raise ValueError("audit actual action metadata disagrees")
        if self.terminated and self.truncated:
            raise ValueError("an audit transition cannot terminate and truncate")
        if self.new_q != self.old_q + self.alpha * self.td_error:
            raise ValueError("audit row cannot reconstruct the Q update exactly")
        if self.td_error != self.target - self.old_q:
            raise ValueError("audit TD error cannot be reconstructed exactly")
        if self.target != self.learning_reward + self.bootstrap_value:
            raise ValueError("audit target cannot be reconstructed exactly")


@dataclass(frozen=True, slots=True)
class QLearningResult:
    q_values: np.ndarray
    state_visit_counts: np.ndarray
    state_action_visit_counts: np.ndarray
    valid_state_mask: np.ndarray
    reachable_state_mask: np.ndarray
    terminal_state_mask: np.ndarray
    episode_metrics: tuple[EpisodeMetrics, ...]
    audit_rows: tuple[AuditRow, ...]
    config: QLearningConfig
    seeds: QLearningSeeds
    identity: QLearningRunIdentity
    runtime_seconds: float
    map_checksum: str

    def __post_init__(self) -> None:
        state_shape = self.valid_state_mask.shape
        q_shape = (*state_shape, len(ACTION_ORDER))
        if (state_shape != self.reachable_state_mask.shape
                or state_shape != self.terminal_state_mask.shape
                or len(state_shape) != 3 or state_shape[0] != 2):
            raise ValueError("Q-Learning state masks have invalid shapes")
        if self.q_values.shape != q_shape or self.state_action_visit_counts.shape != q_shape:
            raise ValueError("Q-Learning Q/count arrays have invalid shapes")
        if self.state_visit_counts.shape != state_shape:
            raise ValueError("Q-Learning state count array has invalid shape")
        if self.q_values.dtype != np.float64:
            raise ValueError("Q values must use float64")
        if self.state_visit_counts.dtype != np.int64 or self.state_action_visit_counts.dtype != np.int64:
            raise ValueError("visit counts must use int64")
        if any(mask.dtype != np.bool_ for mask in
               (self.valid_state_mask, self.reachable_state_mask, self.terminal_state_mask)):
            raise ValueError("state masks must use bool dtype")
        if np.any(self.reachable_state_mask & ~self.valid_state_mask):
            raise ValueError("reachable states must be structurally valid")
        unreachable_actions = np.broadcast_to(
            ~self.reachable_state_mask[..., None], self.state_action_visit_counts.shape)
        if np.any(self.state_visit_counts[~self.reachable_state_mask]) or np.any(
                self.state_action_visit_counts[unreachable_actions]):
            raise ValueError("visit counts must be zero outside reachable states")
        if np.any(self.state_action_visit_counts[self.terminal_state_mask]):
            raise ValueError("terminal intended-action counts must be zero")
        if len(self.episode_metrics) != self.config.episodes:
            raise ValueError("episode metric count does not match configuration")
        if any(metric.episode != index for index, metric in enumerate(self.episode_metrics, 1)):
            raise ValueError("episode metrics are not sequential")
        expected_audit_steps = self.episode_metrics[self.config.audit_episode - 1].steps
        if len(self.audit_rows) != expected_audit_steps:
            raise ValueError("audit must capture every step of its configured episode")
        if any(row.episode != self.config.audit_episode for row in self.audit_rows):
            raise ValueError("audit rows belong to the wrong episode")
        total_terminated = sum(metric.terminated for metric in self.episode_metrics)
        if int(self.state_visit_counts[self.terminal_state_mask].sum()) != total_terminated:
            raise ValueError("terminal state visits must equal terminated episodes")
        if not math.isfinite(self.runtime_seconds) or self.runtime_seconds < 0.0:
            raise ValueError("training runtime must be finite and nonnegative")

    def metadata(self) -> dict[str, object]:
        resolved = json.loads(self.identity.config_json)
        student, map_data = resolved["student"], resolved["map"]
        transitions, rewards = resolved["transitions"], resolved["rewards"]
        shaping, learning, seeds = resolved["shaping"], resolved["learning"], resolved["seeds"]
        return {
            "run_id": self.identity.run_id,
            "semantic_config_hash": self.identity.semantic_config_hash,
            "run_config_json": self.identity.config_json,
            "student_id": student["student_id"], "base_seed": student["base_seed"],
            "map_checksum": map_data["checksum"], "rows": map_data["rows"],
            "cols": map_data["cols"], "max_steps": map_data["max_steps"],
            "intended_probability": transitions["intended_probability"],
            "perpendicular_slip_probability": transitions["perpendicular_slip_probability"],
            "reward_step": rewards["step"], "reward_collision": rewards["collision_extra"],
            "reward_penalty": rewards["penalty_extra"], "reward_key": rewards["first_key"],
            "reward_goal": rewards["goal"], "shaping_method": shaping["method"],
            "shaping_version": shaping["version"], "shaping_scale": shaping["scale"],
            "gamma": learning["gamma"], "alpha": learning["alpha"],
            "episodes": learning["episodes"], "epsilon_start": learning["epsilon_start"],
            "epsilon_end": learning["epsilon_end"], "decay_episodes": learning["decay_episodes"],
            "schedule": learning["schedule"], "reward_mode": learning["reward_mode"],
            "audit_episode": learning["audit_episode"], "root_seed": seeds["root"],
            "behavior_seed": seeds["behavior"], "transition_seed": seeds["transition"],
            "seed_derivation": seeds["derivation"],
            "behavior_policy": resolved["behavior_policy"]["identifier"],
            "q_initialization": resolved["q_initialization"]["identifier"],
            "action_order_json": _canonical_json(resolved["actions"]["order"]),
            "runtime_seconds": self.runtime_seconds,
            "total_steps": int(sum(metric.steps for metric in self.episode_metrics)),
            "total_successes": int(sum(metric.success for metric in self.episode_metrics)),
            "total_terminated": int(sum(metric.terminated for metric in self.episode_metrics)),
            "total_truncated": int(sum(metric.truncated for metric in self.episode_metrics)),
            "state_visit_total": int(self.state_visit_counts.sum()),
        }


Seeds = QLearningSeeds
Result = QLearningResult


@dataclass(frozen=True, slots=True)
class QLearningBundlePaths:
    model: Path
    episode_metrics: Path
    audit: Path
    manifest: Path

    def all_paths(self) -> tuple[Path, Path, Path, Path]:
        return (self.model, self.episode_metrics, self.audit, self.manifest)


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                          allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"run configuration is not canonical-JSON compatible: {exc}") from exc


def epsilon_for_episode(config: QLearningConfig, episode: int) -> float:
    if isinstance(episode, bool) or not isinstance(episode, int) or episode <= 0:
        raise ValueError("episode must be a positive integer")
    if episode == 1:
        return config.epsilon_start
    if episode >= config.decay_episodes:
        return config.epsilon_end
    fraction = (episode - 1) / (config.decay_episodes - 1)
    if config.schedule == "linear":
        return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)
    return config.epsilon_start * (config.epsilon_end / config.epsilon_start) ** fraction


def derive_q_learning_seeds(root: int) -> QLearningSeeds:
    if isinstance(root, bool) or not isinstance(root, int) or root < 0:
        raise ValueError("root seed must be a nonnegative integer")
    behavior_sequence, transition_sequence = np.random.SeedSequence(root).spawn(2)
    return QLearningSeeds(
        root=root,
        behavior=int(behavior_sequence.generate_state(1, dtype=np.uint64)[0]),
        transition=int(transition_sequence.generate_state(1, dtype=np.uint64)[0]),
    )


def build_q_learning_run_identity(mdp: MazeMDP, config: QLearningConfig,
                                  seeds: QLearningSeeds) -> QLearningRunIdentity:
    rewards = mdp.rewards
    resolved = {
        "schema_version": RUN_CONFIG_SCHEMA_VERSION,
        "algorithm": ALGORITHM_ID,
        "student": {"student_id": mdp.spec.student_id, "base_seed": mdp.spec.base_seed},
        "map": {"checksum": map_checksum(mdp.spec), "rows": mdp.spec.rows,
                "cols": mdp.spec.cols, "max_steps": mdp.spec.max_steps},
        "transitions": {"intended_probability": mdp.INTENDED_PROBABILITY,
                        "perpendicular_slip_probability": mdp.SLIP_PROBABILITY},
        "rewards": {"step": rewards.step, "collision_extra": rewards.collision,
                    "penalty_extra": rewards.penalty, "first_key": rewards.key,
                    "goal": rewards.goal, "shaping_scale": rewards.shaping_scale},
        "shaping": {"enabled": config.reward_mode == "shaped",
                    "method": config.shaping_method, "version": config.shaping_version,
                    "scale": rewards.shaping_scale},
        "learning": {"gamma": config.gamma, "alpha": config.alpha,
                     "episodes": config.episodes, "schedule": config.schedule,
                     "reward_mode": config.reward_mode,
                     "epsilon_start": config.epsilon_start, "epsilon_end": config.epsilon_end,
                     "decay_episodes": config.decay_episodes,
                     "audit_episode": config.audit_episode},
        "actions": {"order": list(ACTION_NAMES)},
        "seeds": {"root": seeds.root, "behavior": seeds.behavior,
                  "transition": seeds.transition, "derivation": SEED_DERIVATION},
        "behavior_policy": {"identifier": BEHAVIOR_POLICY,
                            "exploration": "uniform_all_actions",
                            "tie_breaking": "uniform_exact_max"},
        "q_initialization": {"identifier": Q_INITIALIZATION, "valid_states": 0.0,
                             "walls": "NaN", "terminal_states": 0.0},
        "episode_semantics": {"termination": "goal_no_bootstrap",
                              "truncation": "max_steps_bootstrap_then_end"},
    }
    config_json = _canonical_json(resolved)
    digest = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    return QLearningRunIdentity(config_json, digest, f"q-learning-{digest}")


def select_epsilon_greedy(action_values: np.ndarray, epsilon: float,
                          rng: np.random.Generator) -> ActionSelection:
    values = np.asarray(action_values, dtype=np.float64)
    if values.shape != (len(ACTION_ORDER),):
        raise ValueError("action_values must have one entry per action")
    if not np.all(np.isfinite(values)):
        raise ValueError("action_values must be finite")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    greedy = tuple(int(index) for index in np.flatnonzero(values == np.max(values)))
    exploring = bool(rng.random() < epsilon)
    action_index = (int(rng.integers(len(ACTION_ORDER))) if exploring
                    else int(rng.choice(np.asarray(greedy, dtype=np.int64))))
    return ActionSelection(ACTION_ORDER[action_index], action_index, epsilon, exploring, greedy)


def apply_q_learning_update(q_values: np.ndarray, *, state: State,
                            intended_action: Action | str, reward: float,
                            next_state: State, alpha: float, gamma: float,
                            terminated: bool, truncated: bool) -> QUpdate:
    if q_values.ndim != 4 or q_values.shape[-1] != len(ACTION_ORDER):
        raise ValueError("q_values must use (key,row,col,action) layout")
    if terminated and truncated:
        raise ValueError("a transition cannot terminate and truncate")
    if not math.isfinite(reward):
        raise ValueError("reward must be finite")
    if not 0.0 < alpha <= 1.0 or not 0.0 <= gamma < 1.0:
        raise ValueError("invalid alpha or gamma")
    action = Action.parse(intended_action)
    action_index = ACTION_INDEX[action]
    source_index = (*state_index(state), action_index)
    old_q = float(q_values[source_index])
    next_row = q_values[state_index(next_state)]
    if not math.isfinite(old_q) or not np.all(np.isfinite(next_row)):
        raise ValueError("Q update references an invalid state")
    next_max_q = float(np.max(next_row))
    bootstrap_value = 0.0 if terminated else gamma * next_max_q
    target = float(reward) + bootstrap_value
    td_error = target - old_q
    new_q = old_q + alpha * td_error
    q_values[source_index] = new_q
    return QUpdate(state, action, action_index, float(reward), next_state, old_q,
                   next_max_q, bootstrap_value, target, td_error, alpha, gamma,
                   new_q, terminated, truncated)


def train_q_learning(mdp: MazeMDP, config: QLearningConfig, *, root_seed: int,
                     identity: QLearningRunIdentity | None = None) -> QLearningResult:
    if config.reward_mode == "shaped":
        if not mdp.use_shaping:
            raise ValueError("shaped reward mode requires an MDP with shaping enabled")
        if mdp.gamma != config.gamma:
            raise ValueError("shaped MDP gamma must match Q-Learning gamma")
    valid, reachable = valid_state_mask(mdp.spec), reachable_state_mask(mdp)
    terminal = terminal_state_mask(mdp.spec)
    q_values = dense_q_array(mdp.spec)
    q_values[valid], q_values[terminal] = 0.0, 0.0
    state_visits = np.zeros(valid.shape, dtype=np.int64)
    state_action_visits = np.zeros(q_values.shape, dtype=np.int64)
    seeds = derive_q_learning_seeds(root_seed)
    resolved_identity = build_q_learning_run_identity(mdp, config, seeds)
    if identity is not None and identity != resolved_identity:
        raise ValueError("precomputed run identity does not match resolved training inputs")
    behavior_rng = np.random.default_rng(seeds.behavior)
    episode = MazeEpisode(mdp, seed=seeds.transition)
    metrics: list[EpisodeMetrics] = []
    audit_rows: list[AuditRow] = []
    training_started = time.perf_counter()
    for episode_number in range(1, config.episodes + 1):
        episode_started = time.perf_counter()
        epsilon = epsilon_for_episode(config, episode_number)
        state = episode.reset()
        state_visits[state_index(state)] += 1
        visited_states, event_counts = {state}, Counter()
        base_return = shaping_return = 0.0
        while not episode.done:
            source = state
            selection = select_epsilon_greedy(q_values[state_index(source)], epsilon, behavior_rng)
            state_action_visits[(*state_index(source), selection.action_index)] += 1
            step = episode.step(selection.action)
            learning_reward = step.base_reward if config.reward_mode == "sparse" else step.total_reward
            update = apply_q_learning_update(
                q_values, state=source, intended_action=selection.action,
                reward=learning_reward, next_state=step.state, alpha=config.alpha,
                gamma=config.gamma, terminated=step.terminated, truncated=step.truncated)
            state = step.state
            state_visits[state_index(state)] += 1
            visited_states.add(state)
            base_return += step.base_reward
            shaping_return += step.shaping_reward
            event_counts.update(step.events)
            if episode_number == config.audit_episode:
                audit_rows.append(AuditRow(
                    episode_number, step.step_number, epsilon, selection.exploring,
                    int(source.has_key), source.row, source.col, selection.action.name,
                    selection.action_index, step.actual_action.name,
                    ACTION_INDEX[step.actual_action], step.probability, int(step.state.has_key),
                    step.state.row, step.state.col,
                    "|".join(event.value for event in step.events), step.base_reward,
                    step.shaping_reward, step.total_reward, learning_reward, update.old_q,
                    update.next_max_q, update.bootstrap_value, update.gamma, update.target,
                    update.td_error, update.alpha, update.new_q, step.terminated, step.truncated))
        total_return = base_return + shaping_return
        metrics.append(EpisodeMetrics(
            episode_number, epsilon, episode.elapsed_steps, base_return, shaping_return,
            total_return, base_return if config.reward_mode == "sparse" else total_return,
            event_counts[EventType.GOAL_REACHED] > 0,
            event_counts[EventType.GOAL_REACHED] > 0,
            event_counts[EventType.EPISODE_TRUNCATED] > 0,
            int(event_counts[EventType.MOVE]), int(event_counts[EventType.WALL_COLLISION]),
            int(event_counts[EventType.PENALTY_ENTERED]),
            int(event_counts[EventType.KEY_COLLECTED]),
            int(event_counts[EventType.CLOSED_DOOR_ATTEMPT]),
            int(event_counts[EventType.DOOR_PASSED]), int(event_counts[EventType.TELEPORTED]),
            int(event_counts[EventType.GOAL_REACHED]),
            int(event_counts[EventType.EPISODE_TRUNCATED]), len(visited_states),
            episode.elapsed_steps + 1 - len(visited_states),
            time.perf_counter() - episode_started))
    return QLearningResult(q_values, state_visits, state_action_visits, valid, reachable,
                           terminal, tuple(metrics), tuple(audit_rows), config, seeds,
                           resolved_identity, time.perf_counter() - training_started,
                           map_checksum(mdp.spec))


def _prepare_csv(path: Path | str, overwrite: bool) -> Path:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing CSV: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _csv_value(value: object) -> object:
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, bool):
        return int(value)
    return value


def _csv_provenance(result: QLearningResult) -> dict[str, object]:
    metadata = result.metadata()
    keys = ("run_id", "semantic_config_hash", "student_id", "base_seed", "map_checksum",
            "rows", "cols", "max_steps", "intended_probability",
            "perpendicular_slip_probability", "reward_step", "reward_collision",
            "reward_penalty", "reward_key", "reward_goal", "shaping_method",
            "shaping_version", "shaping_scale", "reward_mode", "gamma", "alpha",
            "episodes", "schedule", "epsilon_start", "epsilon_end", "decay_episodes",
            "audit_episode", "root_seed", "behavior_seed", "transition_seed",
            "seed_derivation", "behavior_policy", "q_initialization", "action_order_json")
    return {key: metadata[key] for key in keys}


def _write_result_csv(path: Path | str, result: QLearningResult, rows: Iterable[object],
                      row_type: type[object], *, overwrite: bool) -> Path:
    destination = _prepare_csv(path, overwrite)
    provenance = _csv_provenance(result)
    row_fields = [field.name for field in fields(row_type)]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*provenance, *row_fields])
        writer.writeheader()
        for row in rows:
            writer.writerow({**{name: _csv_value(value) for name, value in provenance.items()},
                             **{name: _csv_value(getattr(row, name)) for name in row_fields}})
    return destination


def write_episode_metrics_csv(path: Path | str, result: QLearningResult, *,
                              overwrite: bool = False) -> Path:
    return _write_result_csv(path, result, result.episode_metrics, EpisodeMetrics,
                             overwrite=overwrite)


def write_audit_csv(path: Path | str, result: QLearningResult, *,
                    overwrite: bool = False) -> Path:
    return _write_result_csv(path, result, result.audit_rows, AuditRow, overwrite=overwrite)


def preflight_q_learning_bundle(paths: QLearningBundlePaths, *, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths.all_paths() if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing artifact(s): " +
                              ", ".join(str(path) for path in existing))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _count_and_validate_csv(path: Path, *, run_id: str, semantic_config_hash: str,
                            expected_rows: int, kind: str, audit_episode: int) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"run_id", "semantic_config_hash", "episode"}
        if kind == "audit":
            required |= {"step", "old_q", "alpha", "td_error", "new_q", "target",
                         "learning_reward", "bootstrap_value"}
        if reader.fieldnames is None or required - set(reader.fieldnames):
            raise ValueError(f"{kind} CSV is missing required fields")
        count = 0
        for count, row in enumerate(reader, 1):
            if row["run_id"] != run_id or row["semantic_config_hash"] != semantic_config_hash:
                raise ValueError(f"{kind} CSV row provenance mismatch")
            if kind == "episodes" and int(row["episode"]) != count:
                raise ValueError("episode CSV rows must be sequential")
            if kind == "audit":
                if int(row["episode"]) != audit_episode or int(row["step"]) != count:
                    raise ValueError("audit CSV rows must cover one complete sequential episode")
                old_q, alpha = float(row["old_q"]), float(row["alpha"])
                td_error, new_q = float(row["td_error"]), float(row["new_q"])
                target = float(row["target"])
                learning_reward, bootstrap = float(row["learning_reward"]), float(row["bootstrap_value"])
                if new_q != old_q + alpha * td_error:
                    raise ValueError("audit CSV new Q reconstruction failed")
                if td_error != target - old_q:
                    raise ValueError("audit CSV TD-error reconstruction failed")
                if target != learning_reward + bootstrap:
                    raise ValueError("audit CSV target reconstruction failed")
    if count != expected_rows:
        raise ValueError(f"{kind} CSV row count mismatch: expected {expected_rows}, observed {count}")
    return count


def _relative_artifact_path(path: Path, manifest: Path) -> str:
    return Path(os.path.relpath(path, manifest.parent)).as_posix()


def _manifest_document(paths: QLearningBundlePaths, staged_model: Path,
                       staged_metrics: Path, staged_audit: Path,
                       result: QLearningResult) -> dict[str, object]:
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "csv_schema_version": CSV_SCHEMA_VERSION,
        "algorithm": ALGORITHM_ID, "run_id": result.identity.run_id,
        "semantic_config_hash": result.identity.semantic_config_hash,
        "run_config_json": result.identity.config_json, "complete": True,
        "artifacts": {
            "model": {"path": _relative_artifact_path(paths.model, paths.manifest),
                      "sha256": _sha256_file(staged_model), "bytes": staged_model.stat().st_size},
            "episode_metrics": {"path": _relative_artifact_path(paths.episode_metrics, paths.manifest),
                                "sha256": _sha256_file(staged_metrics),
                                "bytes": staged_metrics.stat().st_size,
                                "row_count": len(result.episode_metrics)},
            "audit": {"path": _relative_artifact_path(paths.audit, paths.manifest),
                      "sha256": _sha256_file(staged_audit), "bytes": staged_audit.stat().st_size,
                      "row_count": len(result.audit_rows)}}}


def _staged_path(final: Path, token: str) -> Path:
    return final.with_name(f".{final.stem}.{token}.tmp{final.suffix}")


def _resolve_manifest_artifact(manifest_path: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("manifest artifact path must be a nonempty string")
    return (manifest_path.parent / Path(value)).resolve()


def validate_q_learning_bundle(manifest_path: Path | str, *,
                               expected_spec: MazeSpec | None = None,
                               expected_model: Path | str | None = None
                               ) -> tuple[LoadedQLearning, Mapping[str, Any], QLearningBundlePaths]:
    manifest_source = Path(manifest_path).resolve()
    if not manifest_source.exists():
        raise ValueError(f"Incomplete Q-Learning bundle: manifest is missing: {manifest_source}")
    try:
        document = json.loads(manifest_source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read Q-Learning manifest {manifest_source}: {exc}") from exc
    if not isinstance(document, dict) or document.get("complete") is not True:
        raise ValueError("Incomplete Q-Learning bundle: manifest is not complete")
    if document.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported Q-Learning manifest schema version")
    if document.get("csv_schema_version") != CSV_SCHEMA_VERSION:
        raise ValueError("Unsupported Q-Learning CSV schema version")
    if document.get("algorithm") != ALGORITHM_ID:
        raise ValueError("Q-Learning manifest algorithm mismatch")
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Q-Learning manifest artifacts must be an object")
    try:
        model_entry, metrics_entry, audit_entry = (artifacts["model"],
                                                    artifacts["episode_metrics"], artifacts["audit"])
    except KeyError as exc:
        raise ValueError(f"Q-Learning manifest is missing artifact {exc.args[0]!r}") from exc
    if not all(isinstance(entry, dict) for entry in (model_entry, metrics_entry, audit_entry)):
        raise ValueError("Q-Learning manifest artifact entries must be objects")
    paths = QLearningBundlePaths(
        _resolve_manifest_artifact(manifest_source, model_entry.get("path")),
        _resolve_manifest_artifact(manifest_source, metrics_entry.get("path")),
        _resolve_manifest_artifact(manifest_source, audit_entry.get("path")), manifest_source)
    if expected_model is not None and paths.model != Path(expected_model).resolve():
        raise ValueError("Q-Learning manifest references a different model path")
    for name, path, entry in (("model", paths.model, model_entry),
                              ("episode_metrics", paths.episode_metrics, metrics_entry),
                              ("audit", paths.audit, audit_entry)):
        if not path.exists():
            raise ValueError(f"Incomplete Q-Learning bundle: {name} artifact is missing: {path}")
        if entry.get("sha256") != _sha256_file(path):
            raise ValueError(f"Q-Learning {name} artifact hash mismatch")
        if int(entry.get("bytes", -1)) != path.stat().st_size:
            raise ValueError(f"Q-Learning {name} artifact size mismatch")
    loaded = load_q_learning_npz(paths.model, expected_spec=expected_spec)
    metadata = loaded.metadata
    if document.get("run_id") != metadata["run_id"]:
        raise ValueError("manifest/model run_id mismatch")
    if document.get("semantic_config_hash") != metadata["semantic_config_hash"]:
        raise ValueError("manifest/model semantic config hash mismatch")
    if document.get("run_config_json") != metadata["run_config_json"]:
        raise ValueError("manifest/model resolved configuration mismatch")
    episode_rows = _count_and_validate_csv(
        paths.episode_metrics, run_id=str(metadata["run_id"]),
        semantic_config_hash=str(metadata["semantic_config_hash"]),
        expected_rows=int(metadata["episodes"]), kind="episodes",
        audit_episode=int(metadata["audit_episode"]))
    audit_rows = _count_and_validate_csv(
        paths.audit, run_id=str(metadata["run_id"]),
        semantic_config_hash=str(metadata["semantic_config_hash"]),
        expected_rows=int(audit_entry.get("row_count", -1)), kind="audit",
        audit_episode=int(metadata["audit_episode"]))
    if int(metrics_entry.get("row_count", -1)) != episode_rows:
        raise ValueError("manifest episode row count mismatch")
    if int(audit_entry.get("row_count", -1)) != audit_rows:
        raise ValueError("manifest audit row count mismatch")
    return loaded, document, paths


def save_q_learning_bundle(paths: QLearningBundlePaths, result: QLearningResult, *,
                           expected_spec: MazeSpec, overwrite: bool = False
                           ) -> tuple[LoadedQLearning, Mapping[str, Any]]:
    preflight_q_learning_bundle(paths, overwrite=overwrite)
    for destination in paths.all_paths():
        destination.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    staged = QLearningBundlePaths(*(_staged_path(path, token) for path in paths.all_paths()))
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        save_q_learning_npz(
            staged.model, q_values=result.q_values,
            state_visit_counts=result.state_visit_counts,
            state_action_visit_counts=result.state_action_visit_counts,
            valid_mask=result.valid_state_mask, reachable_mask=result.reachable_state_mask,
            terminal_mask=result.terminal_state_mask, metadata=result.metadata())
        write_episode_metrics_csv(staged.episode_metrics, result)
        write_audit_csv(staged.audit, result)
        load_q_learning_npz(staged.model, expected_spec=expected_spec)
        _count_and_validate_csv(staged.episode_metrics, run_id=result.identity.run_id,
                                semantic_config_hash=result.identity.semantic_config_hash,
                                expected_rows=len(result.episode_metrics), kind="episodes",
                                audit_episode=result.config.audit_episode)
        _count_and_validate_csv(staged.audit, run_id=result.identity.run_id,
                                semantic_config_hash=result.identity.semantic_config_hash,
                                expected_rows=len(result.audit_rows), kind="audit",
                                audit_episode=result.config.audit_episode)
        manifest = _manifest_document(paths, staged.model, staged.episode_metrics,
                                      staged.audit, result)
        staged.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True,
                                              ensure_ascii=True) + "\n", encoding="utf-8")
        json.loads(staged.manifest.read_text(encoding="utf-8"))
        if not overwrite:
            preflight_q_learning_bundle(paths, overwrite=False)
        for final in paths.all_paths():
            if final.exists():
                backup = _staged_path(final, f"{token}.backup")
                os.replace(final, backup)
                backups[final] = backup
        for temporary, final in ((staged.model, paths.model),
                                 (staged.episode_metrics, paths.episode_metrics),
                                 (staged.audit, paths.audit),
                                 (staged.manifest, paths.manifest)):
            os.replace(temporary, final)
            published.append(final)
        loaded, validated_manifest, _ = validate_q_learning_bundle(
            paths.manifest, expected_spec=expected_spec, expected_model=paths.model)
    except Exception:
        for final in reversed(published):
            try:
                final.unlink(missing_ok=True)
            except OSError:
                pass
        for final, backup in reversed(tuple(backups.items())):
            if backup.exists():
                os.replace(backup, final)
        raise
    else:
        for backup in backups.values():
            backup.unlink(missing_ok=True)
        return loaded, validated_manifest
    finally:
        for temporary in staged.all_paths():
            temporary.unlink(missing_ok=True)
