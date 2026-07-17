"""Shared dense-state, policy-mask, checksum, and NPZ persistence helpers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from environments.generator import source_map_document
from environments.maze import Action, MazeMDP, MazeSpec, State

ACTION_ORDER: tuple[Action, ...] = tuple(Action)
ACTION_NAMES: tuple[str, ...] = tuple(action.name for action in ACTION_ORDER)
FORMAT_VERSION = 1
ALGORITHM_NAME = "value_iteration"


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
        "state_layout": np.asarray("key,row,col; action last"),
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
    if str(_scalar(data, "state_layout")) != "key,row,col; action last":
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
