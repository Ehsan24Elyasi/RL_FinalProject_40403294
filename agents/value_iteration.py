"""Manual synchronous Value Iteration over MazeMDP.transition_outcomes()."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from environments.maze import MazeMDP, State
from .common import (
    ACTION_ORDER,
    dense_q_array,
    dense_value_array,
    map_checksum,
    optimal_action_masks,
    reachable_state_mask,
    state_index,
    terminal_state_mask,
    valid_state_mask,
)


@dataclass(frozen=True, slots=True)
class ValueIterationConfig:
    gamma: float
    reward_mode: str
    theta: float = 1e-10
    max_sweeps: int = 100_000
    tie_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must be in [0, 1)")
        if self.reward_mode not in {"sparse", "shaped"}:
            raise ValueError("reward_mode must be 'sparse' or 'shaped'")
        if self.theta <= 0.0:
            raise ValueError("theta must be positive")
        if self.max_sweeps <= 0:
            raise ValueError("max_sweeps must be positive")
        if self.tie_tolerance < 0.0:
            raise ValueError("tie_tolerance must be nonnegative")


@dataclass(frozen=True, slots=True)
class ValueIterationResult:
    values: np.ndarray
    q_values: np.ndarray
    optimal_action_mask: np.ndarray
    valid_state_mask: np.ndarray
    reachable_state_mask: np.ndarray
    terminal_state_mask: np.ndarray
    delta_history: np.ndarray
    config: ValueIterationConfig
    iterations: int
    runtime_seconds: float
    bellman_residual: float
    map_checksum: str

    def metadata(self) -> dict[str, object]:
        return {
            "map_checksum": self.map_checksum,
            "gamma": self.config.gamma,
            "theta": self.config.theta,
            "tie_tolerance": self.config.tie_tolerance,
            "reward_mode": self.config.reward_mode,
            "iterations": self.iterations,
            "final_delta": float(self.delta_history[-1]),
            "bellman_residual": self.bellman_residual,
            "runtime_seconds": self.runtime_seconds,
            "converged": True,
            "rows": self.values.shape[1],
            "cols": self.values.shape[2],
        }


class ValueIterationConvergenceError(RuntimeError):
    def __init__(self, max_sweeps: int, final_delta: float) -> None:
        self.max_sweeps = max_sweeps
        self.final_delta = final_delta
        super().__init__(
            f"Value Iteration did not converge within {max_sweeps} sweeps "
            f"(final delta={final_delta:.6g})"
        )


def _reward(outcome: object, reward_mode: str) -> float:
    return float(
        outcome.base_reward if reward_mode == "sparse" else outcome.total_reward
    )


def _state_q_values(
    mdp: MazeMDP,
    state: State,
    values: np.ndarray,
    config: ValueIterationConfig,
) -> np.ndarray:
    action_values = np.empty(len(ACTION_ORDER), dtype=np.float64)
    for action_index, action in enumerate(ACTION_ORDER):
        total = 0.0
        for outcome in mdp.transition_outcomes(state, action):
            continuation = 0.0
            if not outcome.terminated:
                continuation = config.gamma * values[state_index(outcome.state)]
            total += outcome.probability * (
                _reward(outcome, config.reward_mode) + continuation
            )
        action_values[action_index] = total
    return action_values


def value_iteration(mdp: MazeMDP, config: ValueIterationConfig) -> ValueIterationResult:
    """Solve every structurally valid state with synchronous Bellman sweeps."""

    if config.reward_mode == "shaped":
        if not mdp.use_shaping:
            raise ValueError("shaped reward mode requires an MDP with shaping enabled")
        if mdp.gamma != config.gamma:
            raise ValueError("shaped MDP gamma must match Value Iteration gamma")
    elif mdp.use_shaping:
        # Sparse mode deliberately consumes base_reward even if shaping is available.
        pass

    valid = valid_state_mask(mdp.spec)
    terminal = terminal_state_mask(mdp.spec)
    reachable = reachable_state_mask(mdp)
    active = valid & ~terminal
    values = dense_value_array(mdp.spec)
    values[valid] = 0.0
    history: list[float] = []
    started = time.perf_counter()

    for _ in range(config.max_sweeps):
        updated = values.copy()
        for key_index, row, col in zip(*np.nonzero(active), strict=True):
            state = State(row, col, bool(key_index))
            updated[key_index, row, col] = np.max(
                _state_q_values(mdp, state, values, config)
            )
        updated[terminal] = 0.0
        delta = float(np.max(np.abs(updated[active] - values[active])))
        history.append(delta)
        values = updated
        if delta <= config.theta:
            break
    else:
        raise ValueIterationConvergenceError(config.max_sweeps, history[-1])

    q_values = dense_q_array(mdp.spec)
    q_values[terminal] = 0.0
    residual = 0.0
    for key_index, row, col in zip(*np.nonzero(active), strict=True):
        state = State(row, col, bool(key_index))
        row_q = _state_q_values(mdp, state, values, config)
        q_values[key_index, row, col] = row_q
        residual = max(
            residual,
            abs(float(values[key_index, row, col]) - float(np.max(row_q))),
        )
    masks = optimal_action_masks(
        q_values,
        valid,
        terminal,
        tolerance=config.tie_tolerance,
    )
    return ValueIterationResult(
        values=values,
        q_values=q_values,
        optimal_action_mask=masks,
        valid_state_mask=valid,
        reachable_state_mask=reachable,
        terminal_state_mask=terminal,
        delta_history=np.asarray(history, dtype=np.float64),
        config=config,
        iterations=len(history),
        runtime_seconds=time.perf_counter() - started,
        bellman_residual=residual,
        map_checksum=map_checksum(mdp.spec),
    )


def compare_policy_invariance(
    sparse: ValueIterationResult,
    shaped: ValueIterationResult,
) -> tuple[bool, np.ndarray]:
    """Compare complete optimal-action sets on reachable nonterminal states."""

    if sparse.map_checksum != shaped.map_checksum:
        raise ValueError("Cannot compare policies from different maps")
    if sparse.config.gamma != shaped.config.gamma:
        raise ValueError("Cannot compare policies with different gamma values")
    comparison = (
        sparse.reachable_state_mask
        & shaped.reachable_state_mask
        & ~sparse.terminal_state_mask
        & ~shaped.terminal_state_mask
    )
    disagreements = comparison & np.any(
        sparse.optimal_action_mask != shaped.optimal_action_mask, axis=-1
    )
    return not bool(np.any(disagreements)), disagreements
