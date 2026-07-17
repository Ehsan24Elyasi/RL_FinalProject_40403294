"""Phase 2 tests for configuration, dense helpers, VI, persistence, and CLI."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from agents.common import (
    ACTION_ORDER,
    load_value_iteration_npz,
    optimal_action_masks,
    reachable_state_mask,
    save_value_iteration_npz,
    state_from_index,
    state_index,
    terminal_state_mask,
    valid_state_mask,
)
from agents.value_iteration import (
    ValueIterationConfig,
    ValueIterationConvergenceError,
    compare_policy_invariance,
    value_iteration,
)
from config import load_config
from environments.generator import DEFAULT_MAP_PATH, load_source_map
from environments.maze import Action, MazeMDP, MazeSpec, RewardSpec, State

ROOT = Path(__file__).parents[1]


def _boundary(size: int) -> frozenset[tuple[int, int]]:
    return frozenset(
        (row, col)
        for row in range(size)
        for col in range(size)
        if row in (0, size - 1) or col in (0, size - 1)
    )


def _tiny_spec() -> MazeSpec:
    return MazeSpec(
        rows=6,
        cols=6,
        start=(1, 1),
        goal=(1, 4),
        key=(3, 1),
        door=(2, 3),
        walls=_boundary(6),
        penalties=frozenset({(2, 1)}),
        teleporter_pair=((3, 2), (4, 4)),
    )


def test_state_helpers_masks_and_reachability() -> None:
    spec = _tiny_spec()
    mdp = MazeMDP(spec)
    state = State(2, 1, True)
    assert ACTION_ORDER == (Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT)
    assert state_from_index(state_index(state)) == state
    valid = valid_state_mask(spec)
    terminal = terminal_state_mask(spec)
    reachable = reachable_state_mask(mdp)
    assert valid.shape == (2, 6, 6)
    assert not valid[:, 0, 0].any()
    assert terminal[:, 1, 4].all()
    assert reachable[state_index(mdp.initial_state())]
    # Reachability follows the actual transition graph, including all slips.
    assert reachable[0, 1, 4]
    assert not reachable[0, 3, 1]  # Entering the key cell immediately sets has_key.


def test_optimal_masks_use_absolute_tolerance_and_exclude_terminal() -> None:
    q = np.full((2, 2, 2, 4), np.nan)
    solved = np.zeros((2, 2, 2), dtype=bool)
    terminal = np.zeros_like(solved)
    solved[0, 0, 0] = True
    solved[0, 0, 1] = True
    terminal[0, 0, 1] = True
    q[0, 0, 0] = [1.0, 1.0 + 5e-11, 0.0, 0.0]
    q[0, 0, 1] = 0.0
    masks = optimal_action_masks(q, solved, terminal, tolerance=1e-10)
    assert masks[0, 0, 0].tolist() == [True, True, False, False]
    assert not masks[0, 0, 1].any()


def test_vi_first_sweep_is_synchronous_and_weighted() -> None:
    spec = _tiny_spec()
    rewards = RewardSpec(step=-1.0, collision=0.0, penalty=0.0, key=0.0, goal=0.0)
    mdp = MazeMDP(spec, rewards)
    with pytest.raises(ValueIterationConvergenceError) as error:
        value_iteration(
            mdp,
            ValueIterationConfig(0.5, "sparse", theta=1e-30, max_sweeps=1),
        )
    assert error.value.final_delta == pytest.approx(1.0)


def test_vi_terminal_no_bootstrap_collision_and_reward_selection() -> None:
    spec = _tiny_spec()
    sparse_mdp = MazeMDP(spec, gamma=0.5)
    sparse = value_iteration(
        sparse_mdp,
        ValueIterationConfig(0.5, "sparse", theta=1e-8, max_sweeps=1000),
    )
    terminal_index = state_index(State(*spec.goal, has_key=True))
    assert sparse.values[terminal_index] == 0.0
    assert np.all(sparse.q_values[terminal_index] == 0.0)
    assert not sparse.optimal_action_mask[terminal_index].any()
    source = State(1, 3, True)
    action_index = ACTION_ORDER.index(Action.RIGHT)
    expected = 0.0
    for outcome in sparse_mdp.transition_outcomes(source, Action.RIGHT):
        continuation = 0.0 if outcome.terminated else 0.5 * sparse.values[state_index(outcome.state)]
        expected += outcome.probability * (outcome.base_reward + continuation)
    assert sparse.q_values[(*state_index(source), action_index)] == pytest.approx(expected)

    shaped_mdp = MazeMDP(
        spec,
        gamma=0.5,
        use_shaping=True,
        potential_fn=lambda state: float(state.row),
    )
    shaped = value_iteration(
        shaped_mdp,
        ValueIterationConfig(0.5, "shaped", theta=1e-8, max_sweeps=1000),
    )
    sparse_from_shaped = value_iteration(
        shaped_mdp,
        ValueIterationConfig(0.5, "sparse", theta=1e-8, max_sweeps=1000),
    )
    assert not np.allclose(shaped.q_values[valid_state_mask(spec)], sparse.q_values[valid_state_mask(spec)])
    assert np.allclose(sparse_from_shaped.values, sparse.values, equal_nan=True)


def test_vi_validation_and_gamma_mismatch() -> None:
    spec = _tiny_spec()
    with pytest.raises(ValueError, match="gamma"):
        ValueIterationConfig(1.0, "sparse")
    with pytest.raises(ValueError, match="theta"):
        ValueIterationConfig(0.9, "sparse", theta=0.0)
    with pytest.raises(ValueError, match="shaping enabled"):
        value_iteration(MazeMDP(spec), ValueIterationConfig(0.9, "shaped"))
    with pytest.raises(ValueError, match="must match"):
        value_iteration(
            MazeMDP(spec, gamma=0.8, use_shaping=True),
            ValueIterationConfig(0.9, "shaped"),
        )


def test_npz_round_trip_and_mismatch_rejection(tmp_path: Path) -> None:
    spec = _tiny_spec()
    result = value_iteration(
        MazeMDP(spec), ValueIterationConfig(0.5, "sparse", theta=1e-8)
    )
    path = tmp_path / "model.npz"
    save_value_iteration_npz(
        path,
        values=result.values,
        q_values=result.q_values,
        optimal_action_mask=result.optimal_action_mask,
        valid_mask=result.valid_state_mask,
        reachable_mask=result.reachable_state_mask,
        terminal_mask=result.terminal_state_mask,
        delta_history=result.delta_history,
        metadata=result.metadata(),
    )
    loaded = load_value_iteration_npz(path, expected_spec=spec)
    assert np.array_equal(loaded.values, result.values, equal_nan=True)
    with pytest.raises(FileExistsError):
        save_value_iteration_npz(
            path,
            values=result.values,
            q_values=result.q_values,
            optimal_action_mask=result.optimal_action_mask,
            valid_mask=result.valid_state_mask,
            reachable_mask=result.reachable_state_mask,
            terminal_mask=result.terminal_state_mask,
            delta_history=result.delta_history,
            metadata=result.metadata(),
        )
    other = MazeSpec(
        rows=6,
        cols=6,
        start=(1, 1),
        goal=(1, 3),
        key=(3, 1),
        door=(2, 3),
        walls=_boundary(6),
        penalties=frozenset({(2, 1)}),
        teleporter_pair=((3, 2), (4, 4)),
    )
    with pytest.raises(ValueError, match="checksum"):
        load_value_iteration_npz(path, expected_spec=other)


def test_config_validation_and_relative_paths(tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["maze"]["source_map"] = "map.json"
    raw["planning"]["output_dir"] = "models"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(path)
    assert config.source_map == (tmp_path / "map.json").resolve()
    assert config.planning.output_dir == (tmp_path / "models").resolve()
    assert config.rewards.collision == -0.9
    raw["maze"]["intended_probability"] = 0.7
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="0.8/0.1/0.1"):
        load_config(path)


def test_canonical_convergence_invariance_and_cli_inspect(tmp_path: Path) -> None:
    spec = load_source_map(DEFAULT_MAP_PATH)
    sparse = value_iteration(
        MazeMDP(spec, gamma=0.95),
        ValueIterationConfig(0.95, "sparse", theta=1e-10, max_sweeps=100_000),
    )
    shaped = value_iteration(
        MazeMDP(spec, gamma=0.95, use_shaping=True),
        ValueIterationConfig(0.95, "shaped", theta=1e-10, max_sweeps=100_000),
    )
    invariant, disagreements = compare_policy_invariance(sparse, shaped)
    assert sparse.bellman_residual <= 1.1e-10
    assert shaped.bellman_residual <= 1.1e-10
    assert invariant
    assert not disagreements.any()

    path = tmp_path / "canonical.npz"
    save_value_iteration_npz(
        path,
        values=sparse.values,
        q_values=sparse.q_values,
        optimal_action_mask=sparse.optimal_action_mask,
        valid_mask=sparse.valid_state_mask,
        reachable_mask=sparse.reachable_state_mask,
        terminal_mask=sparse.terminal_state_mask,
        delta_history=sparse.delta_history,
        metadata=sparse.metadata(),
    )
    completed = subprocess.run(
        [sys.executable, "main.py", "vi", "inspect", str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Bellman residual" in completed.stdout
