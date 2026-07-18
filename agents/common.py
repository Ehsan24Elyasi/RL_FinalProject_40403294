"""Shared dense-state, policy-mask, checksum, and NPZ persistence helpers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from environments.generator import source_map_document
from environments.maze import Action, MazeMDP, MazeSpec, State

ACTION_ORDER: tuple[Action, ...] = tuple(Action)
ACTION_NAMES: tuple[str, ...] = tuple(action.name for action in ACTION_ORDER)
FORMAT_VERSION = 1
ALGORITHM_NAME = "value_iteration"
Q_LEARNING_FORMAT_VERSION = 1
Q_LEARNING_ALGORITHM_NAME = "q_learning"
STATE_LAYOUT = "key,row,col; action last"


def state_index(state: State) -> tuple[int, int, int]:
    return (int(state.has_key), state.row, state.col)


def state_from_index(index: tuple[int, int, int]) -> State:
    key_index, row, col = index
    if key_index not in (0, 1):
        raise ValueError("key index must be 0 or 1")
    return State(row=row, col=col, has_key=bool(key_index))


def valid_state_mask(spec: MazeSpec) -> np.ndarray:
    mask = np.ones((2, spec.rows, spec.cols), dtype=np.bool_)
    for row, col in spec.walls:
        mask[:, row, col] = False
    return mask


def terminal_state_mask(spec: MazeSpec) -> np.ndarray:
    mask = np.zeros((2, spec.rows, spec.cols), dtype=np.bool_)
    mask[:, spec.goal[0], spec.goal[1]] = True
    return mask


def reachable_state_mask(mdp: MazeMDP) -> np.ndarray:
    """Traverse every positive-probability branch without expanding terminals."""

    mask = np.zeros((2, mdp.spec.rows, mdp.spec.cols), dtype=np.bool_)
    start = mdp.initial_state()
    queue: deque[State] = deque([start])
    mask[state_index(start)] = True
    while queue:
        state = queue.popleft()
        if state.position == mdp.spec.goal:
            continue
        for action in ACTION_ORDER:
            for outcome in mdp.transition_outcomes(state, action):
                if outcome.probability <= 0.0:
                    continue
                index = state_index(outcome.state)
                if not mask[index]:
                    mask[index] = True
                    queue.append(outcome.state)
    return mask


def dense_value_array(spec: MazeSpec) -> np.ndarray:
    return np.full((2, spec.rows, spec.cols), np.nan, dtype=np.float64)


def dense_q_array(spec: MazeSpec) -> np.ndarray:
    return np.full((2, spec.rows, spec.cols, len(ACTION_ORDER)), np.nan, dtype=np.float64)


def optimal_action_masks(
    q_values: np.ndarray,
    solved_mask: np.ndarray,
    terminal_mask: np.ndarray,
    *,
    tolerance: float,
) -> np.ndarray:
    if tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative")
    expected_shape = (*solved_mask.shape, len(ACTION_ORDER))
    if q_values.shape != expected_shape:
        raise ValueError(f"q_values must have shape {expected_shape}")
    masks = np.zeros(q_values.shape, dtype=np.bool_)
    active = solved_mask & ~terminal_mask
    for index in zip(*np.nonzero(active), strict=True):
        row = q_values[index]
        best = np.max(row)
        masks[index] = np.isclose(row, best, rtol=0.0, atol=tolerance)
    return masks


def map_checksum(spec: MazeSpec) -> str:
    return str(source_map_document(spec)["checksum"]["value"])


@dataclass(frozen=True, slots=True)
class LoadedValueIteration:
    values: np.ndarray
    q_values: np.ndarray
    optimal_action_mask: np.ndarray
    valid_state_mask: np.ndarray
    reachable_state_mask: np.ndarray
    terminal_state_mask: np.ndarray
    delta_history: np.ndarray
    metadata: Mapping[str, Any]


def _scalar(data: Mapping[str, np.ndarray], key: str) -> Any:
    if key not in data:
        raise ValueError(f"NPZ is missing required field {key!r}")
    array = data[key]
    if array.shape != ():
        raise ValueError(f"NPZ field {key!r} must be scalar")
    return array.item()


def save_value_iteration_npz(
    path: Path | str,
    *,
    values: np.ndarray,
    q_values: np.ndarray,
    optimal_action_mask: np.ndarray,
    valid_mask: np.ndarray,
    reachable_mask: np.ndarray,
    terminal_mask: np.ndarray,
    delta_history: np.ndarray,
    metadata: Mapping[str, Any],
    overwrite: bool = False,
) -> Path:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing model: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "values": np.asarray(values, dtype=np.float64),
        "q_values": np.asarray(q_values, dtype=np.float64),
        "optimal_action_mask": np.asarray(optimal_action_mask, dtype=np.bool_),
        "valid_state_mask": np.asarray(valid_mask, dtype=np.bool_),
        "reachable_state_mask": np.asarray(reachable_mask, dtype=np.bool_),
        "terminal_state_mask": np.asarray(terminal_mask, dtype=np.bool_),
        "delta_history": np.asarray(delta_history, dtype=np.float64),
        "format_version": np.asarray(FORMAT_VERSION, dtype=np.int64),
        "algorithm": np.asarray(ALGORITHM_NAME),
        "action_names": np.asarray(ACTION_NAMES),
        "state_layout": np.asarray(STATE_LAYOUT),
    }
    for key, value in metadata.items():
        if key in payload:
            raise ValueError(f"metadata key conflicts with reserved NPZ field: {key}")
        payload[key] = np.asarray(value)
    np.savez_compressed(destination, **payload)
    return destination


def load_value_iteration_npz(
    path: Path | str,
    *,
    expected_spec: MazeSpec | None = None,
) -> LoadedValueIteration:
    source = Path(path)
    try:
        archive_context = np.load(source, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load model {source}: {exc}") from exc
    with archive_context as archive:
        data = {key: archive[key] for key in archive.files}

    required_arrays = {
        "values",
        "q_values",
        "optimal_action_mask",
        "valid_state_mask",
        "reachable_state_mask",
        "terminal_state_mask",
        "delta_history",
        "action_names",
    }
    missing = required_arrays - data.keys()
    if missing:
        raise ValueError(f"NPZ is missing required fields: {sorted(missing)}")
    if int(_scalar(data, "format_version")) != FORMAT_VERSION:
        raise ValueError("Unsupported NPZ format version")
    if str(_scalar(data, "algorithm")) != ALGORITHM_NAME:
        raise ValueError("NPZ algorithm metadata mismatch")
    if tuple(str(name) for name in data["action_names"].tolist()) != ACTION_NAMES:
        raise ValueError("NPZ action ordering mismatch")
    if str(_scalar(data, "state_layout")) != STATE_LAYOUT:
        raise ValueError("NPZ state layout mismatch")

    values = data["values"]
    q_values = data["q_values"]
    optimal = data["optimal_action_mask"]
    valid = data["valid_state_mask"]
    reachable = data["reachable_state_mask"]
    terminal = data["terminal_state_mask"]
    history = data["delta_history"]
    if values.dtype != np.float64 or q_values.dtype != np.float64 or history.dtype != np.float64:
        raise ValueError("NPZ value, Q, and history arrays must be float64")
    if optimal.dtype != np.bool_ or valid.dtype != np.bool_ or reachable.dtype != np.bool_ or terminal.dtype != np.bool_:
        raise ValueError("NPZ masks must use bool dtype")
    if values.ndim != 3 or values.shape[0] != 2:
        raise ValueError("NPZ values shape must be (2, rows, cols)")
    expected_q_shape = (*values.shape, len(ACTION_ORDER))
    if q_values.shape != expected_q_shape or optimal.shape != expected_q_shape:
        raise ValueError("NPZ Q/action-mask shape mismatch")
    if valid.shape != values.shape or reachable.shape != values.shape or terminal.shape != values.shape:
        raise ValueError("NPZ state-mask shape mismatch")
    if history.ndim != 1 or history.size == 0 or not np.all(np.isfinite(history)):
        raise ValueError("NPZ delta history must be nonempty and finite")
    if np.any(reachable & ~valid) or np.any(terminal & ~valid):
        raise ValueError("NPZ masks are inconsistent")
    solved = valid
    if not np.all(np.isfinite(values[solved])) or not np.all(np.isnan(values[~solved])):
        raise ValueError("NPZ values have invalid finite/NaN placement")
    q_solved = np.broadcast_to(solved[..., None], q_values.shape)
    if not np.all(np.isfinite(q_values[q_solved])) or not np.all(np.isnan(q_values[~q_solved])):
        raise ValueError("NPZ Q values have invalid finite/NaN placement")
    if np.any(optimal[terminal]) or np.any(optimal[~solved]):
        raise ValueError("NPZ optimal masks must be empty outside nonterminal solved states")
    if not np.all(values[terminal] == 0.0) or not np.all(q_values[terminal] == 0.0):
        raise ValueError("NPZ terminal values and Q values must be zero")

    scalar_keys = (
        "map_checksum",
        "gamma",
        "theta",
        "tie_tolerance",
        "reward_mode",
        "iterations",
        "final_delta",
        "bellman_residual",
        "runtime_seconds",
        "converged",
        "rows",
        "cols",
    )
    metadata = {key: _scalar(data, key) for key in scalar_keys}
    if not bool(metadata["converged"]):
        raise ValueError("NPZ does not represent a converged solution")
    if int(metadata["iterations"]) != history.size:
        raise ValueError("NPZ iteration count does not match history length")
    if float(metadata["final_delta"]) != float(history[-1]):
        raise ValueError("NPZ final delta does not match history")
    if int(metadata["rows"]) != values.shape[1] or int(metadata["cols"]) != values.shape[2]:
        raise ValueError("NPZ dimensions metadata mismatch")
    if str(metadata["reward_mode"]) not in {"sparse", "shaped"}:
        raise ValueError("NPZ reward mode is invalid")
    if expected_spec is not None:
        if values.shape != (2, expected_spec.rows, expected_spec.cols):
            raise ValueError("NPZ shape does not match expected map")
        if str(metadata["map_checksum"]) != map_checksum(expected_spec):
            raise ValueError("NPZ map checksum mismatch")
        if not np.array_equal(valid, valid_state_mask(expected_spec)):
            raise ValueError("NPZ valid-state mask does not match expected map")
        if not np.array_equal(terminal, terminal_state_mask(expected_spec)):
            raise ValueError("NPZ terminal-state mask does not match expected map")

    return LoadedValueIteration(values, q_values, optimal, valid, reachable, terminal, history, metadata)


@dataclass(frozen=True, slots=True)
class LoadedQLearning:
    q_values: np.ndarray
    state_visit_counts: np.ndarray
    state_action_visit_counts: np.ndarray
    valid_state_mask: np.ndarray
    reachable_state_mask: np.ndarray
    terminal_state_mask: np.ndarray
    metadata: Mapping[str, Any]


def save_q_learning_npz(
    path: Path | str,
    *,
    q_values: np.ndarray,
    state_visit_counts: np.ndarray,
    state_action_visit_counts: np.ndarray,
    valid_mask: np.ndarray,
    reachable_mask: np.ndarray,
    terminal_mask: np.ndarray,
    metadata: Mapping[str, Any],
    overwrite: bool = False,
) -> Path:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing model: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "q_values": np.asarray(q_values, dtype=np.float64),
        "state_visit_counts": np.asarray(state_visit_counts, dtype=np.int64),
        "state_action_visit_counts": np.asarray(state_action_visit_counts, dtype=np.int64),
        "valid_state_mask": np.asarray(valid_mask, dtype=np.bool_),
        "reachable_state_mask": np.asarray(reachable_mask, dtype=np.bool_),
        "terminal_state_mask": np.asarray(terminal_mask, dtype=np.bool_),
        "format_version": np.asarray(Q_LEARNING_FORMAT_VERSION, dtype=np.int64),
        "algorithm": np.asarray(Q_LEARNING_ALGORITHM_NAME),
        "action_names": np.asarray(ACTION_NAMES),
        "state_layout": np.asarray(STATE_LAYOUT),
    }
    for key, value in metadata.items():
        if key in payload:
            raise ValueError(f"metadata key conflicts with reserved NPZ field: {key}")
        payload[key] = np.asarray(value)
    np.savez_compressed(destination, **payload)
    return destination


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def load_q_learning_npz(
    path: Path | str,
    *,
    expected_spec: MazeSpec | None = None,
) -> LoadedQLearning:
    source = Path(path)
    try:
        archive_context = np.load(source, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load model {source}: {exc}") from exc
    with archive_context as archive:
        data = {key: archive[key] for key in archive.files}

    required_arrays = {
        "q_values", "state_visit_counts", "state_action_visit_counts",
        "valid_state_mask", "reachable_state_mask", "terminal_state_mask",
        "action_names",
    }
    missing = required_arrays - data.keys()
    if missing:
        raise ValueError(f"NPZ is missing required fields: {sorted(missing)}")
    if int(_scalar(data, "format_version")) != Q_LEARNING_FORMAT_VERSION:
        raise ValueError("Unsupported Q-Learning NPZ format version")
    if str(_scalar(data, "algorithm")) != Q_LEARNING_ALGORITHM_NAME:
        raise ValueError("NPZ algorithm metadata mismatch")
    if tuple(str(name) for name in data["action_names"].tolist()) != ACTION_NAMES:
        raise ValueError("NPZ action ordering mismatch")
    if str(_scalar(data, "state_layout")) != STATE_LAYOUT:
        raise ValueError("NPZ state layout mismatch")

    q_values = data["q_values"]
    state_visits = data["state_visit_counts"]
    state_action_visits = data["state_action_visit_counts"]
    valid, reachable = data["valid_state_mask"], data["reachable_state_mask"]
    terminal = data["terminal_state_mask"]
    if q_values.dtype != np.float64:
        raise ValueError("NPZ Q values must use float64")
    if state_visits.dtype != np.int64 or state_action_visits.dtype != np.int64:
        raise ValueError("NPZ visit counts must use int64")
    if any(mask.dtype != np.bool_ for mask in (valid, reachable, terminal)):
        raise ValueError("NPZ masks must use bool dtype")
    if q_values.ndim != 4 or q_values.shape[0] != 2 or q_values.shape[-1] != len(ACTION_ORDER):
        raise ValueError("NPZ Q values shape must be (2, rows, cols, 4)")
    state_shape = q_values.shape[:-1]
    if any(array.shape != state_shape for array in (state_visits, valid, reachable, terminal)):
        raise ValueError("NPZ state arrays have inconsistent shapes")
    if state_action_visits.shape != q_values.shape:
        raise ValueError("NPZ state-action count shape mismatch")
    if np.any(state_visits < 0) or np.any(state_action_visits < 0):
        raise ValueError("NPZ visit counts must be nonnegative")
    if np.any(reachable & ~valid) or np.any(terminal & ~valid):
        raise ValueError("NPZ masks are inconsistent")
    valid_q = np.broadcast_to(valid[..., None], q_values.shape)
    if not np.all(np.isfinite(q_values[valid_q])) or not np.all(np.isnan(q_values[~valid_q])):
        raise ValueError("NPZ Q values have invalid finite/NaN placement")
    invalid_q = np.broadcast_to(~valid[..., None], state_action_visits.shape)
    if np.any(state_visits[~valid]) or np.any(state_action_visits[invalid_q]):
        raise ValueError("NPZ invalid states must have zero visit counts")
    unreachable_q = np.broadcast_to(~reachable[..., None], state_action_visits.shape)
    if np.any(state_visits[~reachable]) or np.any(state_action_visits[unreachable_q]):
        raise ValueError("NPZ counts must be zero outside reachable states")
    if not np.all(q_values[terminal] == 0.0):
        raise ValueError("NPZ terminal Q values must be zero")
    if np.any(state_action_visits[terminal]):
        raise ValueError("NPZ terminal states must have zero intended-action counts")

    scalar_keys = (
        "run_id", "semantic_config_hash", "run_config_json", "student_id", "base_seed",
        "map_checksum", "rows", "cols", "max_steps", "intended_probability",
        "perpendicular_slip_probability", "reward_step", "reward_collision",
        "reward_penalty", "reward_key", "reward_goal", "shaping_method",
        "shaping_version", "shaping_scale", "gamma", "alpha", "episodes",
        "epsilon_start", "epsilon_end", "decay_episodes", "schedule", "reward_mode",
        "audit_episode", "root_seed", "behavior_seed", "transition_seed",
        "seed_derivation", "behavior_policy", "q_initialization", "action_order_json",
        "runtime_seconds", "total_steps", "total_successes", "total_terminated",
        "total_truncated", "state_visit_total",
    )
    metadata = {key: _scalar(data, key) for key in scalar_keys}
    try:
        resolved = json.loads(str(metadata["run_config_json"]))
    except json.JSONDecodeError as exc:
        raise ValueError("NPZ resolved run configuration JSON is invalid") from exc
    canonical = _canonical_json(resolved)
    if canonical != str(metadata["run_config_json"]):
        raise ValueError("NPZ resolved run configuration is not canonical JSON")
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if str(metadata["semantic_config_hash"]) != digest:
        raise ValueError("NPZ semantic config hash mismatch")
    if str(metadata["run_id"]) != f"q-learning-{digest}":
        raise ValueError("NPZ run_id mismatch")
    try:
        if resolved["schema_version"] != 1:
            raise ValueError("NPZ run configuration schema version mismatch")
        if resolved["algorithm"] != "manual_off_policy_q_learning":
            raise ValueError("NPZ run configuration algorithm mismatch")
        student, map_data, transitions = resolved["student"], resolved["map"], resolved["transitions"]
        rewards, shaping = resolved["rewards"], resolved["shaping"]
        learning, seeds = resolved["learning"], resolved["seeds"]
        actions = resolved["actions"]
        if transitions["intended_probability"] != MazeMDP.INTENDED_PROBABILITY:
            raise ValueError("NPZ intended transition probability mismatch")
        if transitions["perpendicular_slip_probability"] != MazeMDP.SLIP_PROBABILITY:
            raise ValueError("NPZ slip transition probability mismatch")
        if rewards["shaping_scale"] != shaping["scale"]:
            raise ValueError("NPZ shaping scale fields disagree")
        if shaping["method"] != "normalized_completion_distance" or shaping["version"] != 1:
            raise ValueError("NPZ shaping method/version is unsupported")
        if shaping["enabled"] != (learning["reward_mode"] == "shaped"):
            raise ValueError("NPZ shaping-enabled and reward-mode fields disagree")
        if seeds["derivation"] != "numpy.SeedSequence(root).spawn(2); uint64 child state":
            raise ValueError("NPZ seed derivation is unsupported")
        if resolved["behavior_policy"]["identifier"] != "epsilon_greedy_uniform_exploration_uniform_exact_max_ties":
            raise ValueError("NPZ behavior policy is unsupported")
        if resolved["q_initialization"]["identifier"] != "valid_zero_walls_nan_terminal_zero":
            raise ValueError("NPZ Q initialization is unsupported")
        if resolved["episode_semantics"] != {
            "termination": "goal_no_bootstrap",
            "truncation": "max_steps_bootstrap_then_end",
        }:
            raise ValueError("NPZ episode semantics are unsupported")
        expected_fields = {
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
            "action_order_json": _canonical_json(actions["order"]),
        }
    except (KeyError, TypeError) as exc:
        raise ValueError("NPZ resolved run configuration is missing required fields") from exc
    for key, expected in expected_fields.items():
        actual = metadata[key]
        if isinstance(expected, float):
            if float(actual) != expected:
                raise ValueError(f"NPZ metadata field {key!r} disagrees with run configuration")
        elif str(actual) != str(expected):
            raise ValueError(f"NPZ metadata field {key!r} disagrees with run configuration")
    if tuple(actions["order"]) != ACTION_NAMES:
        raise ValueError("NPZ run configuration action order mismatch")
    if not 0.0 <= float(metadata["gamma"]) < 1.0 or not 0.0 < float(metadata["alpha"]) <= 1.0:
        raise ValueError("NPZ learning-rate metadata is invalid")
    episodes, decay_episodes = int(metadata["episodes"]), int(metadata["decay_episodes"])
    audit_episode = int(metadata["audit_episode"])
    if episodes <= 0 or decay_episodes < 2 or not 1 <= audit_episode <= episodes:
        raise ValueError("NPZ episode metadata is invalid")
    epsilon_start, epsilon_end = float(metadata["epsilon_start"]), float(metadata["epsilon_end"])
    if not 0.0 <= epsilon_end <= epsilon_start <= 1.0:
        raise ValueError("NPZ epsilon metadata is invalid")
    if str(metadata["schedule"]) not in {"linear", "exponential", "geometric"}:
        raise ValueError("NPZ epsilon schedule is invalid")
    if str(metadata["reward_mode"]) not in {"sparse", "shaped"}:
        raise ValueError("NPZ reward mode is invalid")
    if int(metadata["rows"]) != state_shape[1] or int(metadata["cols"]) != state_shape[2]:
        raise ValueError("NPZ dimensions metadata mismatch")
    if any(int(metadata[key]) < 0 for key in ("root_seed", "behavior_seed", "transition_seed")):
        raise ValueError("NPZ seed metadata is invalid")
    if int(metadata["behavior_seed"]) == int(metadata["transition_seed"]):
        raise ValueError("NPZ behavior and transition seeds are not independent")
    total_steps, total_terminated = int(metadata["total_steps"]), int(metadata["total_terminated"])
    total_successes, total_truncated = int(metadata["total_successes"]), int(metadata["total_truncated"])
    if total_steps != int(state_action_visits.sum()):
        raise ValueError("NPZ total step count does not match intended-action counts")
    state_visit_total = int(state_visits.sum())
    if int(metadata["state_visit_total"]) != state_visit_total or state_visit_total != total_steps + episodes:
        raise ValueError("NPZ state visit total does not match counts")
    if total_successes != total_terminated or total_terminated + total_truncated != episodes:
        raise ValueError("NPZ completed-episode totals disagree")
    if int(state_visits[terminal].sum()) != total_terminated:
        raise ValueError("NPZ terminal state visits must equal terminated episodes")
    if not np.isfinite(float(metadata["runtime_seconds"])) or float(metadata["runtime_seconds"]) < 0.0:
        raise ValueError("NPZ runtime metadata is invalid")

    if expected_spec is not None:
        if state_shape != (2, expected_spec.rows, expected_spec.cols):
            raise ValueError("NPZ shape does not match expected map")
        if str(metadata["student_id"]) != expected_spec.student_id or int(metadata["base_seed"]) != expected_spec.base_seed:
            raise ValueError("NPZ student/map identity mismatch")
        if str(metadata["map_checksum"]) != map_checksum(expected_spec):
            raise ValueError("NPZ map checksum mismatch")
        if int(metadata["max_steps"]) != expected_spec.max_steps:
            raise ValueError("NPZ max_steps metadata mismatch")
        expected_valid, expected_terminal = valid_state_mask(expected_spec), terminal_state_mask(expected_spec)
        expected_reachable = reachable_state_mask(MazeMDP(expected_spec))
        if not np.array_equal(valid, expected_valid):
            raise ValueError("NPZ valid-state mask does not match expected map")
        if not np.array_equal(terminal, expected_terminal):
            raise ValueError("NPZ terminal-state mask does not match expected map")
        if not np.array_equal(reachable, expected_reachable):
            raise ValueError("NPZ reachable-state mask does not match expected map")

    return LoadedQLearning(q_values, state_visits, state_action_visits, valid,
                           reachable, terminal, metadata)
