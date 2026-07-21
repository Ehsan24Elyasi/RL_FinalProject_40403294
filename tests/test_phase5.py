from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from agents.common import ACTION_ORDER, dense_q_array
from config import load_config
from environments.generator import load_source_map
from environments.maze import EventType, MazeMDP
from experiments import (
    ATTEMPT_SCHEMA_VERSION,
    BUILTIN_PRESETS,
    CSV_SCHEMA_VERSION,
    FINAL_PRESETS,
    MANIFEST_SCHEMA_VERSION,
    PRESET_SCHEMA_VERSION,
    EvaluationSpec,
    Preset,
    SeedStreams,
    audit_event_coverage,
    audit_events,
    common_rollout_seeds,
    derive_seed_streams,
    evaluate_q_values,
    expand_preset,
    filter_runs,
    list_presets,
    load_preset,
    run_preset,
    status_preset,
    validate_attempt,
)
from experiments import runner
from experiments.run_experiments import build_parser, main

ROOT = Path(__file__).parents[1]
CONFIG_DIR = ROOT / "experiments" / "configs"
EXPECTED = {
    "smoke": (4, 3, 4, 8, 11, 8, 19),
    "final-vi": (4, 0, 0, 0, 4, 0, 4),
    "final-q": (0, 15, 0, 0, 15, 0, 15),
    "final-sarsa": (0, 0, 20, 0, 20, 0, 20),
    "final-transfer": (0, 0, 0, 40, 0, 40, 40),
    "final-all": (4, 15, 20, 40, 39, 40, 79),
}


def canonical_mdp(*, shaping=True):
    cfg = load_config(ROOT / "config.yaml")
    spec = load_source_map(cfg.source_map)
    return MazeMDP(spec, cfg.rewards, gamma=.95, use_shaping=shaping)


def write_mutated_preset(tmp_path, mutate):
    document = yaml.safe_load((CONFIG_DIR / "smoke.yaml").read_text(encoding="utf-8"))
    mutate(document)
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def tiny_vi_preset(tmp_path):
    base = load_preset("smoke", output_root=tmp_path)
    vi = next(slot for slot in base.slots if slot.key == "vi-shaped-gamma-0.95")
    return Preset(base.name, base.source, base.project_config, tmp_path.resolve(), False,
                  EvaluationSpec(1, False, "phase5-test"), (vi,), False)


def normalized_evaluation(result):
    return (
        tuple(replace(row, runtime_seconds=0) for row in result.episodes),
        result.steps,
        result.tie_seeds,
        result.transition_seeds,
    )


def test_public_schema_constants_and_preset_inventory():
    assert (PRESET_SCHEMA_VERSION, ATTEMPT_SCHEMA_VERSION,
            MANIFEST_SCHEMA_VERSION, CSV_SCHEMA_VERSION) == (1, 1, 1, 1)
    assert list_presets() == BUILTIN_PRESETS == (
        "smoke", "final-vi", "final-q", "final-sarsa", "final-transfer", "final-all")
    assert FINAL_PRESETS == frozenset(BUILTIN_PRESETS[1:])


@pytest.mark.parametrize("name", BUILTIN_PRESETS)
def test_all_presets_have_locked_schema_and_counts(name):
    preset = load_preset(name)
    runs = expand_preset(preset)
    counts = (
        sum(run.algorithm == "vi" for run in runs),
        sum(run.algorithm == "q_learning" for run in runs),
        sum(run.algorithm == "sarsa_lambda" for run in runs),
        sum(run.algorithm == "transfer" for run in runs),
        sum(run.available for run in runs),
        sum(not run.available for run in runs),
        len(runs),
    )
    assert counts == EXPECTED[name]
    assert preset.final == (name in FINAL_PRESETS)
    assert all(run.seeds is None for run in runs if run.algorithm == "vi")
    assert all(run.seeds is not None for run in runs if run.root_seed is not None)
    assert all(not run.available and run.unavailable_reason
               for run in runs if run.algorithm == "transfer")


def test_final_matrix_variants_roots_and_filters():
    runs = expand_preset(load_preset("final-all"))
    roots = (9, 109, 209, 309, 409)
    q = [run for run in runs if run.algorithm == "q_learning"]
    sarsa = [run for run in runs if run.algorithm == "sarsa_lambda"]
    transfer = [run for run in runs if run.algorithm == "transfer"]
    assert {run.slot_key for run in q} == {
        "q-shaped-linear", "q-shaped-exponential", "q-sparse-linear"}
    assert {run.root_seed for run in q} == set(roots)
    assert {run.parameters["trace_lambda"] for run in sarsa} == {0., .3, .7, .9}
    assert {(run.parameters["transfer_role"], run.parameters["transfer_variant"])
            for run in transfer} == {
                (role, variant) for role in ("similar", "different")
                for variant in ("scratch", "full", "scaled", "selective")}
    assert len(filter_runs(runs, slots=["q-shaped-linear"], seeds=[109])) == 1
    assert filter_runs(runs, seeds=[999]) == ()
    assert filter_runs(runs, slots=["vi-shaped-gamma-0.95"], seeds=[9]) == ()
    with pytest.raises(ValueError, match="unknown slot"):
        filter_runs(runs, slots=["not-a-slot"])


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d.__setitem__("unknown", 1), "unknown keys"),
    (lambda d: d.__setitem__("schema_version", 99), "schema_version"),
    (lambda d: d.pop("matrix"), "missing keys"),
    (lambda d: d["seeds"].__setitem__("roots", 9), "must be a list"),
    (lambda d: d["seeds"].__setitem__("roots", [True]), "must be an integer"),
    (lambda d: d["seeds"].__setitem__("stream_derivation", "other"), "seed derivation"),
    (lambda d: d["seeds"].__setitem__("stream_order", []), "stream order"),
    (lambda d: d["evaluation"].__setitem__("policy", "epsilon-greedy"), "evaluation semantics"),
    (lambda d: d["evaluation"].__setitem__("preserve_arrays", False), "evaluation semantics"),
    (lambda d: d["expected_counts"].__setitem__("q_runs", 999), "expected count mismatch"),
    (lambda d: d["matrix"]["value_iteration"]["variants"].append(
        dict(d["matrix"]["value_iteration"]["variants"][0])), "duplicate slot keys"),
    (lambda d: d["matrix"]["transfer_q_learning"]["defaults"].__setitem__("status", "ready"),
     "must be unavailable"),
])
def test_strict_preset_rejections(tmp_path, mutate, match):
    with pytest.raises(ValueError, match=match):
        load_preset(write_mutated_preset(tmp_path, mutate))


def test_five_seed_streams_are_exact_deterministic_and_validated():
    streams = derive_seed_streams(9)
    expected = tuple(int(child.generate_state(1, dtype=np.uint64)[0])
                     for child in np.random.SeedSequence(9).spawn(5))
    assert tuple(asdict(streams)[name] for name in runner.SEED_STREAM_ORDER) == expected
    assert derive_seed_streams(9) == streams
    assert len(set(expected)) == 5
    assert all(type(value) is int and value >= 0 for value in (streams.root, *expected))
    for invalid in (-1, True, 1.5):
        with pytest.raises(ValueError):
            derive_seed_streams(invalid)
    with pytest.raises(ValueError, match="derivation/order"):
        SeedStreams(streams.root, streams.training_action, streams.training_action,
                    streams.evaluation_tie_breaking, streams.evaluation_transition,
                    streams.statistics)
    with pytest.raises(ValueError, match="derivation/order"):
        replace(streams, order=tuple(reversed(streams.order)))


def test_common_rollout_seeds_group_root_and_fallback_reproducibility():
    spec = EvaluationSpec(4, False, "paired")
    streams = derive_seed_streams(9)
    first = common_rollout_seeds(streams, spec)
    assert first == common_rollout_seeds(streams, spec)
    assert first != common_rollout_seeds(streams, replace(spec, group="other"))
    assert first != common_rollout_seeds(derive_seed_streams(109), spec)
    assert first[0] != first[1]
    assert common_rollout_seeds(None, spec, fallback_root=9) == first


def test_evaluation_is_exact_tie_reproducible_and_non_mutating():
    mdp = canonical_mdp()
    q = np.zeros((2, mdp.spec.rows, mdp.spec.cols, len(ACTION_ORDER)), dtype=float)
    state = mdp.initial_state()
    q[(int(state.has_key), state.row, state.col)] = [2., 2., 1.999999999999, -5.]
    q_before = q.copy()
    evaluation = EvaluationSpec(3, True, "exact-ties")
    first = evaluate_q_values(mdp, q, evaluation, streams=derive_seed_streams(9),
                              learning_reward_mode="shaped")
    second = evaluate_q_values(mdp, q, evaluation, streams=derive_seed_streams(9),
                               learning_reward_mode="shaped")
    assert np.array_equal(q, q_before, equal_nan=True)
    assert normalized_evaluation(first) == normalized_evaluation(second)
    initial_choices = [step.intended_action_index for step in first.steps
                       if step.step == 1]
    assert initial_choices and set(initial_choices) <= {0, 1}
    for episode in first.episodes:
        assert episode.total_return == pytest.approx(episode.base_return + episode.shaping_return)
        assert episode.learning_return == pytest.approx(episode.total_return)
        assert episode.unique_state_visits + episode.repeated_state_visits == episode.steps + 1
        assert episode.success == episode.terminated == bool(episode.goal_reached_count)
        assert episode.truncated == bool(episode.episode_truncated_count)
    no_steps = evaluate_q_values(mdp, q, EvaluationSpec(1, False, "no-steps"),
                                 fallback_root=9, learning_reward_mode="sparse")
    assert no_steps.steps == ()
    assert no_steps.episodes[0].learning_return == pytest.approx(no_steps.episodes[0].base_return)


def test_evaluation_shape_and_nonfinite_guards():
    mdp = canonical_mdp()
    with pytest.raises(ValueError, match="shape"):
        evaluate_q_values(mdp, np.zeros((1,)), EvaluationSpec(1, False))
    q = dense_q_array(mdp.spec)
    q[np.isfinite(q)] = 0.
    state = mdp.initial_state()
    q[(int(state.has_key), state.row, state.col, 0)] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_q_values(mdp, q, EvaluationSpec(1, False))


def test_all_nine_events_have_real_witnesses_and_audit_outputs(tmp_path):
    witnesses = audit_event_coverage(canonical_mdp())
    assert tuple(w.event for w in witnesses) == tuple(event.value for event in EventType)
    assert len(witnesses) == 9
    assert witnesses[-1].source == "MazeEpisode(max_steps=1)"
    assert all(w.intended_action and w.actual_action and 0 < w.outcome_probability <= 1
               for w in witnesses)
    written = audit_events("smoke", output_root=tmp_path)
    assert written == witnesses
    assert (tmp_path / "smoke" / "event_coverage.csv").is_file()
    manifest = json.loads((tmp_path / "smoke" / "preset_manifest.json").read_text())
    assert manifest["event_audit"] == {
        "required": [event.value for event in EventType],
        "covered": [event.value for event in EventType],
        "missing": [],
    }


def test_cli_parser_list_resilience_and_final_guard(monkeypatch, tmp_path, capsys):
    parsed = build_parser().parse_args([
        "run", "smoke", "--slot", "q-shaped-linear", "--seed", "9",
        "--no-resume", "--keep-going", "--require-clean-git"])
    assert parsed.slot == ["q-shaped-linear"] and parsed.seed == [9]
    assert not parsed.resume and parsed.keep_going and parsed.require_clean_git

    original = runner.status_preset
    monkeypatch.setattr(runner, "status_preset", lambda name, **kwargs:
                        (_ for _ in ()).throw(ValueError("broken")) if name == "final-q"
                        else original(name, **kwargs))
    assert main(["list", "--output-root", str(tmp_path)]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 6
    failed = next(row for row in rows if row["preset"] == "final-q")
    assert failed["status"] == "failed" and "broken" in failed["error"]

    assert main(["run", "final-vi", "--output-root", str(tmp_path)]) == 2
    assert "--confirm-final" in capsys.readouterr().err


def test_clean_git_guard_rejects_dirty_or_unavailable(monkeypatch, tmp_path):
    preset = tiny_vi_preset(tmp_path)
    for dirty in (True, None):
        monkeypatch.setattr(runner, "repository_provenance", lambda **kwargs: {
            "git": {"commit": "x", "dirty": dirty, "dirty_fingerprint": "f"}})
        with pytest.raises(RuntimeError, match="clean Git"):
            run_preset(preset, require_clean_git=True)


def test_attempt_success_resume_retry_failure_and_corruption(tmp_path, monkeypatch):
    preset = tiny_vi_preset(tmp_path)
    provenance = {"git": {"commit": "test", "dirty": False,
                           "dirty_fingerprint": "clean"}}
    monkeypatch.setattr(runner, "repository_provenance", lambda **kwargs: provenance)

    first = run_preset(preset)
    assert (first.complete, first.skipped, first.failed, first.selected) == (1, 0, 0, 1)
    run = expand_preset(preset)[0]
    slot_dir = tmp_path / "smoke" / "attempts" / run.slot_id
    attempts = sorted(slot_dir.glob("attempt-*"))
    assert len(attempts) == 1
    ledger = validate_attempt(attempts[0])
    assert ledger["status"] == ledger["stage"] == "complete"
    assert {"resolved_config.json", "map.json", "attempt.json", "model.npz", "episodes.csv"} <= {
        path.name for path in attempts[0].iterdir()}
    assert ledger["artifacts"]["episodes"]["row_count"] == 1
    assert len(ledger["evaluation_seed_vectors"]["tie_breaking"]) == 1

    resumed = run_preset(preset)
    assert (resumed.complete, resumed.skipped) == (0, 1)
    retried = run_preset(preset, resume=False)
    assert retried.complete == 1 and len(list(slot_dir.glob("attempt-*"))) == 2

    with pytest.raises(RuntimeError, match="controlled training failure"):
        run_preset(preset, resume=False, forced_failure_stage="training")
    attempts = sorted(slot_dir.glob("attempt-*"))
    failed_ledger = json.loads((attempts[-1] / "attempt.json").read_text())
    assert failed_ledger["status"] == "failed"
    assert failed_ledger["failure"]["type"] == "RuntimeError"
    assert "traceback" in failed_ledger["failure"]
    with pytest.raises(ValueError, match="not complete"):
        validate_attempt(attempts[-1])

    model = attempts[0] / ledger["artifacts"]["model"]["path"]
    model.write_bytes(model.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="artifact validation failed"):
        validate_attempt(attempts[0])
    summary = run_preset(preset)
    assert summary.skipped == 1  # the second, still-valid complete attempt satisfies resume

    manifest = status_preset(preset)
    assert manifest["planned"] == 1 and manifest["failed"] == 1
