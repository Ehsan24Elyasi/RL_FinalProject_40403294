"""Phase 3 tests for Q-Learning numerics, reproducibility, persistence, and CLI."""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

import agents.q_learning as q_module
from agents.common import (
    ACTION_ORDER,
    load_q_learning_npz,
    load_value_iteration_npz,
    save_q_learning_npz,
    state_index,
)
from agents.q_learning import (
    QLearningBundlePaths,
    QLearningConfig,
    apply_q_learning_update,
    build_q_learning_run_identity,
    derive_q_learning_seeds,
    epsilon_for_episode,
    save_q_learning_bundle,
    select_epsilon_greedy,
    train_q_learning,
    validate_q_learning_bundle,
    write_audit_csv,
    write_episode_metrics_csv,
)
from config import QLearningRun, load_config
from environments.generator import DEFAULT_MAP_PATH, load_source_map
from environments.maze import (
    Action,
    EventType,
    MazeMDP,
    MazeSpec,
    RewardSpec,
    State,
    StepResult,
)

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


def _result(episodes: int = 3, *, reward_mode: str = "sparse", schedule: str = "linear"):
    spec = _tiny_spec()
    mdp = MazeMDP(
        spec,
        gamma=0.95,
        use_shaping=reward_mode == "shaped",
    )
    config = QLearningConfig(
        episodes=episodes,
        decay_episodes=max(2, episodes),
        audit_episode=1,
        reward_mode=reward_mode,
        schedule=schedule,
    )
    return train_q_learning(mdp, config, root_seed=9)


def _save_result(path: Path, result, *, overwrite: bool = False) -> Path:
    return save_q_learning_npz(
        path,
        q_values=result.q_values,
        state_visit_counts=result.state_visit_counts,
        state_action_visit_counts=result.state_action_visit_counts,
        valid_mask=result.valid_state_mask,
        reachable_mask=result.reachable_state_mask,
        terminal_mask=result.terminal_state_mask,
        metadata=result.metadata(),
        overwrite=overwrite,
    )


def _without_runtime(metric) -> dict[str, object]:
    row = asdict(metric)
    row.pop("runtime_seconds")
    return row


def test_q_learning_config_and_artifact_stem_validation() -> None:
    assert QLearningConfig() == QLearningConfig(
        gamma=0.95,
        alpha=0.1,
        episodes=5000,
        epsilon_start=1.0,
        epsilon_end=0.05,
        decay_episodes=4000,
        schedule="linear",
        reward_mode="shaped",
        audit_episode=1,
    )
    assert QLearningRun("shaped", "exponential", 9).artifact_stem == (
        "q_shaped_exponential_seed_9_ep_5000"
    )
    with pytest.raises(ValueError, match="alpha"):
        QLearningConfig(alpha=0.0)
    with pytest.raises(ValueError, match="audit"):
        QLearningConfig(episodes=2, audit_episode=3)
    with pytest.raises(ValueError, match="positive"):
        QLearningRun("shaped", "linear", 9, 0)


def test_linear_and_exponential_schedules_have_exact_endpoints() -> None:
    linear = QLearningConfig(schedule="linear")
    assert epsilon_for_episode(linear, 1) == 1.0
    assert epsilon_for_episode(linear, 4000) == 0.05
    assert epsilon_for_episode(linear, 4001) == 0.05
    assert epsilon_for_episode(linear, 2000) == pytest.approx(
        1.0 + (1999 / 3999) * (0.05 - 1.0)
    )

    exponential = QLearningConfig(schedule="exponential")
    geometric = QLearningConfig(schedule="geometric")
    for config in (exponential, geometric):
        assert epsilon_for_episode(config, 1) == 1.0
        assert epsilon_for_episode(config, 4000) == 0.05
        assert epsilon_for_episode(config, 4001) == 0.05
        assert epsilon_for_episode(config, 2000) == pytest.approx(
            1.0 * (0.05 / 1.0) ** (1999 / 3999)
        )
    with pytest.raises(ValueError, match="positive"):
        epsilon_for_episode(linear, 0)


def test_seed_derivation_matches_seedsequence_and_is_independent() -> None:
    seeds = derive_q_learning_seeds(9)
    expected = np.random.SeedSequence(9).spawn(2)
    assert seeds.behavior == int(expected[0].generate_state(1, dtype=np.uint64)[0])
    assert seeds.transition == int(expected[1].generate_state(1, dtype=np.uint64)[0])
    assert seeds.behavior != seeds.transition
    assert derive_q_learning_seeds(9) == seeds


def test_epsilon_greedy_uniform_exploration_and_exact_max_ties() -> None:
    action_values = np.array([2.0, 2.0, 1.999999999999, -5.0], dtype=np.float64)
    exploit_rng = np.random.default_rng(123)
    exploit_counts = np.zeros(4, dtype=int)
    for _ in range(10_000):
        selection = select_epsilon_greedy(action_values, 0.0, exploit_rng)
        exploit_counts[selection.action_index] += 1
        assert selection.greedy_action_indices == (0, 1)
        assert not selection.exploring
    assert exploit_counts[2] == exploit_counts[3] == 0
    assert abs(exploit_counts[0] - exploit_counts[1]) < 400

    explore_rng = np.random.default_rng(456)
    explore_counts = np.zeros(4, dtype=int)
    for _ in range(20_000):
        selection = select_epsilon_greedy(action_values, 1.0, explore_rng)
        explore_counts[selection.action_index] += 1
        assert selection.exploring
    assert np.all(np.abs(explore_counts - 5_000) < 300)


def test_q_update_terminal_nonterminal_and_truncated_targets() -> None:
    q = np.zeros((2, 3, 3, 4), dtype=np.float64)
    source = State(1, 1, False)
    next_state = State(1, 2, False)
    q[state_index(next_state)] = [1.0, 3.0, 2.0, -1.0]

    nonterminal = apply_q_learning_update(
        q,
        state=source,
        intended_action=Action.RIGHT,
        reward=2.0,
        next_state=next_state,
        alpha=0.1,
        gamma=0.95,
        terminated=False,
        truncated=False,
    )
    assert nonterminal.target == pytest.approx(2.0 + 0.95 * 3.0)
    assert nonterminal.new_q == pytest.approx(0.485)

    q[(*state_index(source), ACTION_ORDER.index(Action.UP))] = 4.0
    terminal = apply_q_learning_update(
        q,
        state=source,
        intended_action=Action.UP,
        reward=20.0,
        next_state=next_state,
        alpha=0.1,
        gamma=0.95,
        terminated=True,
        truncated=False,
    )
    assert terminal.bootstrap_value == 0.0
    assert terminal.target == 20.0
    assert terminal.new_q == pytest.approx(5.6)

    truncated = apply_q_learning_update(
        q,
        state=source,
        intended_action=Action.DOWN,
        reward=-0.1,
        next_state=next_state,
        alpha=0.1,
        gamma=0.95,
        terminated=False,
        truncated=True,
    )
    assert truncated.bootstrap_value == pytest.approx(2.85)
    assert truncated.target == pytest.approx(2.75)


def test_training_updates_intended_action_not_sampled_actual(monkeypatch) -> None:
    spec = _tiny_spec()

    class OneSlipEpisode:
        def __init__(self, mdp, *, seed=None, max_steps=None):
            self.mdp = mdp
            self.max_steps = 1
            self._done = False
            self._state = mdp.initial_state()
            self._elapsed_steps = 0

        @property
        def done(self):
            return self._done

        @property
        def elapsed_steps(self):
            return self._elapsed_steps

        def reset(self, *, seed=None):
            assert seed is None
            self._done = False
            self._elapsed_steps = 0
            self._state = self.mdp.initial_state()
            return self._state

        def step(self, action):
            intended = Action.parse(action)
            self._done = True
            self._elapsed_steps = 1
            self._state = State(*spec.goal, has_key=False)
            return StepResult(
                state=self._state,
                intended_action=intended,
                actual_action=Action.DOWN if intended is not Action.DOWN else Action.LEFT,
                probability=0.1,
                events=(EventType.MOVE, EventType.GOAL_REACHED),
                base_reward=20.0,
                shaping_reward=0.0,
                total_reward=20.0,
                terminated=True,
                truncated=False,
                step_number=1,
            )

    monkeypatch.setattr(q_module, "MazeEpisode", OneSlipEpisode)
    result = train_q_learning(
        MazeMDP(spec),
        QLearningConfig(
            episodes=1,
            decay_episodes=2,
            epsilon_start=0.0,
            epsilon_end=0.0,
            reward_mode="sparse",
            audit_episode=1,
        ),
        root_seed=9,
    )
    row = result.audit_rows[0]
    source_index = state_index(State(*spec.start, has_key=False))
    intended_index = row.intended_action_index
    actual_index = row.actual_action_index
    assert intended_index != actual_index
    assert result.q_values[(*source_index, intended_index)] == pytest.approx(2.0)
    assert result.q_values[(*source_index, actual_index)] == 0.0
    assert result.state_action_visit_counts[(*source_index, intended_index)] == 1
    assert result.state_action_visit_counts[(*source_index, actual_index)] == 0


def test_fixed_seed_reproduces_model_counts_metrics_and_audit() -> None:
    first = _result(episodes=4, reward_mode="shaped", schedule="exponential")
    second = _result(episodes=4, reward_mode="shaped", schedule="exponential")
    assert np.array_equal(first.q_values, second.q_values, equal_nan=True)
    assert np.array_equal(first.state_visit_counts, second.state_visit_counts)
    assert np.array_equal(
        first.state_action_visit_counts, second.state_action_visit_counts
    )
    assert [_without_runtime(row) for row in first.episode_metrics] == [
        _without_runtime(row) for row in second.episode_metrics
    ]
    assert first.audit_rows == second.audit_rows


def test_metrics_counts_rewards_visits_and_audit_reconstruct() -> None:
    result = _result(episodes=3, reward_mode="shaped")
    assert int(result.state_action_visit_counts.sum()) == sum(
        metric.steps for metric in result.episode_metrics
    )
    assert int(result.state_visit_counts.sum()) == sum(
        metric.steps + 1 for metric in result.episode_metrics
    )
    for metric in result.episode_metrics:
        assert metric.total_return == pytest.approx(
            metric.base_return + metric.shaping_return
        )
        assert metric.learning_return == metric.total_return
        assert metric.unique_state_visits + metric.repeated_state_visits == metric.steps + 1
        assert metric.terminated ^ metric.truncated
        assert metric.goal_reached_count == int(metric.success)
        assert metric.episode_truncated_count == int(metric.truncated)
    audited_metric = result.episode_metrics[result.config.audit_episode - 1]
    assert len(result.audit_rows) == audited_metric.steps
    assert [row.step for row in result.audit_rows] == list(
        range(1, audited_metric.steps + 1)
    )
    for row in result.audit_rows:
        assert row.new_q == row.old_q + row.alpha * row.td_error
        assert row.td_error == row.target - row.old_q
        expected_bootstrap = 0.0 if row.terminated else row.gamma * row.next_max_q
        assert row.bootstrap_value == expected_bootstrap
        assert row.target == row.learning_reward + row.bootstrap_value
        assert row.total_reward == pytest.approx(row.base_reward + row.shaping_reward)


def test_sparse_learning_return_uses_base_reward() -> None:
    spec = _tiny_spec()
    shaped_mdp = MazeMDP(spec, gamma=0.95, use_shaping=True)
    result = train_q_learning(
        shaped_mdp,
        QLearningConfig(
            episodes=2,
            decay_episodes=2,
            reward_mode="sparse",
            audit_episode=1,
        ),
        root_seed=9,
    )
    assert any(metric.shaping_return != 0.0 for metric in result.episode_metrics)
    assert all(
        metric.learning_return == metric.base_return
        for metric in result.episode_metrics
    )
    assert all(row.learning_reward == row.base_reward for row in result.audit_rows)


def test_q_npz_round_trip_overwrite_and_malformed_rejection(tmp_path: Path) -> None:
    result = _result(episodes=2)
    path = _save_result(tmp_path / "model.npz", result)
    loaded = load_q_learning_npz(path, expected_spec=_tiny_spec())
    assert np.array_equal(loaded.q_values, result.q_values, equal_nan=True)
    assert np.array_equal(loaded.state_visit_counts, result.state_visit_counts)
    with pytest.raises(FileExistsError):
        _save_result(path, result)

    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["action_names"] = np.asarray(["RIGHT", "UP", "DOWN", "LEFT"])
    malformed = tmp_path / "malformed.npz"
    np.savez_compressed(malformed, **payload)
    with pytest.raises(ValueError, match="action ordering"):
        load_q_learning_npz(malformed, expected_spec=_tiny_spec())

    payload["action_names"] = np.asarray([action.name for action in ACTION_ORDER])
    payload["total_steps"] = np.asarray(int(payload["total_steps"]) + 1)
    np.savez_compressed(malformed, **payload)
    with pytest.raises(ValueError, match="total step"):
        load_q_learning_npz(malformed, expected_spec=_tiny_spec())


def test_csv_full_precision_round_trip_and_overwrite(tmp_path: Path) -> None:
    result = _result(episodes=2, reward_mode="shaped")
    metrics_path = write_episode_metrics_csv(tmp_path / "episodes.csv", result)
    audit_path = write_audit_csv(tmp_path / "audit.csv", result)
    with metrics_path.open(encoding="utf-8", newline="") as handle:
        metric_rows = list(csv.DictReader(handle))
    with audit_path.open(encoding="utf-8", newline="") as handle:
        audit_rows = list(csv.DictReader(handle))
    assert len(metric_rows) == 2
    assert len(audit_rows) == result.episode_metrics[0].steps
    row = audit_rows[0]
    old_q = float(row["old_q"])
    alpha = float(row["alpha"])
    td_error = float(row["td_error"])
    assert float(row["new_q"]) == old_q + alpha * td_error
    assert float(row["target"]) == float(row["learning_reward"]) + float(
        row["bootstrap_value"]
    )
    assert all(row["run_id"] == result.identity.run_id for row in metric_rows)
    assert all(row["run_id"] == result.identity.run_id for row in audit_rows)
    with pytest.raises(FileExistsError):
        write_episode_metrics_csv(metrics_path, result)
    with pytest.raises(FileExistsError):
        write_audit_csv(audit_path, result)


def test_config_q_learning_defaults_paths_and_required_runs(tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["maze"]["source_map"] = "map.json"
    raw["q_learning"]["model_dir"] = "models"
    raw["q_learning"]["raw_dir"] = "raw"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(path)
    assert config.q_learning.gamma == 0.95
    assert config.q_learning.alpha == 0.1
    assert config.q_learning.episodes == 5000
    assert config.q_learning.decay_episodes == 4000
    assert config.q_learning.model_dir == (tmp_path / "models").resolve()
    assert config.q_learning.raw_dir == (tmp_path / "raw").resolve()
    assert config.q_learning.required_runs == (
        QLearningRun("shaped", "linear", 9, 5000),
        QLearningRun("shaped", "exponential", 9, 5000),
    )

    raw["q_learning"]["required_runs"][1]["episodes"] = 4999
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="required_runs"):
        load_config(path)


@pytest.mark.parametrize("bad_value", [True, 2.5, "5000", None])
def test_config_rejects_malformed_integer_fields(tmp_path: Path, bad_value) -> None:
    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["q_learning"]["episodes"] = bad_value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="q_learning.episodes must be an integer"):
        load_config(path)


def test_config_requires_explicit_supported_shaping_method(tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["shaping"]["method"] = "unknown"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="shaping.method"):
        load_config(path)


def test_run_identity_changes_for_semantic_configuration_changes() -> None:
    spec = _tiny_spec()
    seeds = derive_q_learning_seeds(9)
    base_mdp = MazeMDP(spec, RewardSpec(), gamma=0.95, use_shaping=True)
    base = QLearningConfig(episodes=2, decay_episodes=2, audit_episode=1)
    identities = {
        build_q_learning_run_identity(base_mdp, base, seeds).run_id,
        build_q_learning_run_identity(
            base_mdp,
            QLearningConfig(episodes=2, decay_episodes=2, audit_episode=1, alpha=0.2),
            seeds,
        ).run_id,
        build_q_learning_run_identity(
            base_mdp,
            QLearningConfig(episodes=2, decay_episodes=3, audit_episode=1),
            seeds,
        ).run_id,
        build_q_learning_run_identity(
            MazeMDP(
                spec,
                RewardSpec(step=-0.2),
                gamma=0.95,
                use_shaping=True,
            ),
            base,
            seeds,
        ).run_id,
    }
    assert len(identities) == 4


def test_bundle_manifest_hashes_rows_and_missing_manifest(tmp_path: Path) -> None:
    result = _result(episodes=2, reward_mode="shaped")
    paths = QLearningBundlePaths(
        tmp_path / "model.npz",
        tmp_path / "episodes.csv",
        tmp_path / "audit.csv",
        tmp_path / "manifest.json",
    )
    loaded, manifest = save_q_learning_bundle(
        paths, result, expected_spec=_tiny_spec()
    )
    assert manifest["complete"] is True
    assert manifest["run_id"] == result.identity.run_id
    assert manifest["artifacts"]["episode_metrics"]["row_count"] == 2
    assert manifest["artifacts"]["audit"]["row_count"] == len(result.audit_rows)
    assert np.array_equal(loaded.reachable_state_mask, result.reachable_state_mask)
    validated, _, validated_paths = validate_q_learning_bundle(
        paths.manifest, expected_spec=_tiny_spec(), expected_model=paths.model
    )
    assert validated.metadata["run_id"] == result.identity.run_id
    paths.manifest.unlink()
    with pytest.raises(ValueError, match="Incomplete.*manifest is missing"):
        validate_q_learning_bundle(validated_paths.manifest, expected_spec=_tiny_spec())


def test_bundle_failure_leaves_no_complete_final_mix(tmp_path: Path, monkeypatch) -> None:
    result = _result(episodes=2, reward_mode="shaped")
    paths = QLearningBundlePaths(
        tmp_path / "model.npz",
        tmp_path / "episodes.csv",
        tmp_path / "audit.csv",
        tmp_path / "manifest.json",
    )
    original = q_module._count_and_validate_csv

    def fail_audit(path, **kwargs):
        if kwargs["kind"] == "audit":
            raise ValueError("injected audit validation failure")
        return original(path, **kwargs)

    monkeypatch.setattr(q_module, "_count_and_validate_csv", fail_audit)
    with pytest.raises(ValueError, match="injected"):
        save_q_learning_bundle(paths, result, expected_spec=_tiny_spec())
    assert not any(path.exists() for path in paths.all_paths())


def test_npz_rejects_unreachable_and_terminal_count_corruption(tmp_path: Path) -> None:
    result = _result(episodes=2)
    path = _save_result(tmp_path / "model.npz", result)
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key].copy() for key in archive.files}
    unreachable_valid = result.valid_state_mask & ~result.reachable_state_mask
    unreachable = tuple(
        int(value) for value in next(zip(*np.nonzero(unreachable_valid), strict=True))
    )
    payload["state_visit_counts"][unreachable] = 1
    payload["state_visit_total"] = np.asarray(int(payload["state_visit_total"]) + 1)
    corrupted = tmp_path / "unreachable.npz"
    np.savez_compressed(corrupted, **payload)
    with pytest.raises(ValueError, match="outside reachable"):
        load_q_learning_npz(corrupted, expected_spec=_tiny_spec())

    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key].copy() for key in archive.files}
    terminal = result.terminal_state_mask
    terminal_visits = int(payload["state_visit_counts"][terminal].sum())
    payload["state_visit_counts"][terminal] = 0
    start = state_index(State(*_tiny_spec().start, has_key=False))
    payload["state_visit_counts"][start] += terminal_visits
    corrupted = tmp_path / "terminal.npz"
    np.savez_compressed(corrupted, **payload)
    with pytest.raises(ValueError, match="terminal state visits"):
        load_q_learning_npz(corrupted, expected_spec=_tiny_spec())


def test_existing_committed_vi_models_remain_loadable() -> None:
    spec = load_source_map(DEFAULT_MAP_PATH)
    paths = sorted((ROOT / "results" / "models" / "value_iteration").glob("*.npz"))
    assert len(paths) == 4
    for path in paths:
        loaded = load_value_iteration_npz(path, expected_spec=spec)
        assert loaded.q_values.shape == (2, 16, 16, 4)


@pytest.mark.parametrize(
    ("reward_mode", "schedule"),
    (("shaped", "linear"), ("shaped", "exponential"), ("sparse", "linear")),
)
def test_canonical_short_coverage(reward_mode: str, schedule: str) -> None:
    spec = load_source_map(DEFAULT_MAP_PATH)
    mdp = MazeMDP(
        spec,
        gamma=0.95,
        use_shaping=reward_mode == "shaped",
    )
    result = train_q_learning(
        mdp,
        QLearningConfig(
            episodes=2,
            decay_episodes=2,
            reward_mode=reward_mode,
            schedule=schedule,
            audit_episode=2,
        ),
        root_seed=9,
    )
    assert result.q_values.shape == (2, 16, 16, 4)
    assert result.reachable_state_mask.shape == (2, 16, 16)
    assert len(result.episode_metrics) == 2
    assert len(result.audit_rows) == result.episode_metrics[1].steps


def test_cli_pilot_inspect_and_overwrite_refusal(tmp_path: Path) -> None:
    command = [
        sys.executable,
        str(ROOT / "main.py"),
        "q",
        "train",
        "--config",
        str(ROOT / "config.yaml"),
        "--reward-mode",
        "shaped",
        "--schedule",
        "linear",
        "--seed",
        "9",
        "--episodes",
        "2",
        "--decay-episodes",
        "2",
        "--audit-episode",
        "1",
        "--output-dir",
        str(tmp_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "Saved and validated bundle model" in completed.stdout
    models = list((tmp_path / "models").glob("q_shaped_linear_seed_9_ep_2_rid_*.npz"))
    manifests = list((tmp_path / "raw").glob("q_shaped_linear_seed_9_ep_2_rid_*_manifest.json"))
    assert len(models) == len(manifests) == 1
    model = models[0]

    inspected = subprocess.run(
        [
            sys.executable,
            str(ROOT / "main.py"),
            "q",
            "inspect",
            str(model),
            "--config",
            str(ROOT / "config.yaml"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert inspected.returncode == 0, inspected.stderr
    assert "Q-Learning model" in inspected.stdout
    assert "episodes=2" in inspected.stdout

    refused = subprocess.run(command, capture_output=True, text=True, check=False)
    assert refused.returncode == 1
    assert "Refusing to overwrite" in refused.stderr
