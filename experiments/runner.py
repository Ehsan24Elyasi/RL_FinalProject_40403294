"""Configuration-driven, reproducible experiment orchestration.

The module intentionally has no CLI dependencies.  ``run_experiments.py`` can
build its commands from the public functions at the bottom of this file.
"""
from __future__ import annotations

import csv
from collections import Counter, deque
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import tracemalloc
from typing import Any, Iterable, Mapping, Sequence
import uuid

import numpy as np
import yaml

from agents.common import (
    ACTION_ORDER, load_q_learning_npz, load_sarsa_lambda_npz,
    load_value_iteration_npz, map_checksum, save_q_learning_npz,
    save_sarsa_lambda_npz, save_value_iteration_npz, state_index,
)
from agents.q_learning import QLearningConfig, QLearningSeeds, train_q_learning
from agents.sarsa_lambda import SarsaLambdaConfig, SarsaLambdaSeeds, train_sarsa_lambda
from agents.value_iteration import ValueIterationConfig, value_iteration
from config import DEFAULT_CONFIG_PATH, OperationalConfig, load_config
from environments.generator import load_source_map, source_map_document
from environments.maze import Action, EventType, MazeEpisode, MazeMDP, MazeSpec, State

PRESET_SCHEMA_VERSION = 1
ATTEMPT_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
CSV_SCHEMA_VERSION = 1
SEED_STREAM_ORDER = ("training_action", "training_transition", "evaluation_tie_breaking", "evaluation_transition", "statistics")
SEED_DERIVATION = "numpy.SeedSequence(root).spawn(5); uint64 child state"
BUILTIN_PRESETS = ("smoke", "final-vi", "final-q", "final-sarsa", "final-transfer", "final-all")
FINAL_PRESETS = frozenset(name for name in BUILTIN_PRESETS if name.startswith("final-"))
TRANSFER_UNAVAILABLE_REASON = "Phase 6 target maps and transfer initialization are not implemented"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRESET_DIR = Path(__file__).with_name("configs")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "experiments"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")


def _strict_keys(value: Any, name: str, allowed: set[str], required: set[str] = frozenset()) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    unknown, missing = set(value) - allowed, required - set(value)
    if unknown:
        raise ValueError(f"{name} has unknown keys: {sorted(unknown)}")
    if missing:
        raise ValueError(f"{name} is missing keys: {sorted(missing)}")
    return value


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _resolved_path(value: Any, name: str, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a nonempty path string")
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


@dataclass(frozen=True, slots=True)
class SeedStreams:
    root: int
    training_action: int
    training_transition: int
    evaluation_tie_breaking: int
    evaluation_transition: int
    statistics: int
    derivation: str = SEED_DERIVATION
    order: tuple[str, ...] = SEED_STREAM_ORDER

    def __post_init__(self) -> None:
        values = (self.root, self.training_action, self.training_transition,
                  self.evaluation_tie_breaking, self.evaluation_transition, self.statistics)
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in values):
            raise ValueError("seed streams must be nonnegative integers")
        if len(set(values[1:])) != 5 or self.order != SEED_STREAM_ORDER or self.derivation != SEED_DERIVATION:
            raise ValueError("seed stream derivation/order is invalid")


def derive_seed_streams(root: int) -> SeedStreams:
    root = _integer(root, "root seed")
    children = np.random.SeedSequence(root).spawn(5)
    values = [int(child.generate_state(1, dtype=np.uint64)[0]) for child in children]
    return SeedStreams(root, *values)


@dataclass(frozen=True, slots=True)
class EvaluationSpec:
    episodes: int = 10
    record_steps: bool = True
    group: str = "common-source-map"

    def __post_init__(self) -> None:
        _integer(self.episodes, "evaluation.episodes", 1)
        if not isinstance(self.record_steps, bool) or not self.group:
            raise ValueError("invalid evaluation settings")


@dataclass(frozen=True, slots=True)
class ScientificSlot:
    key: str
    algorithm: str
    reward_mode: str | None = None
    gamma: float | None = None
    schedule: str | None = None
    trace_lambda: float | None = None
    episodes: int | None = None
    seeds: tuple[int, ...] = ()
    audit_episode: int | None = None
    diagnostic_episode: int | None = None
    transfer_role: str | None = None
    transfer_variant: str | None = None
    available: bool = True
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.key or self.algorithm not in {"vi", "q_learning", "sarsa_lambda", "transfer"}:
            raise ValueError("invalid slot key or algorithm")
        if self.algorithm == "vi":
            if self.reward_mode not in {"sparse", "shaped"} or self.gamma is None or self.seeds:
                raise ValueError("VI slots require reward_mode/gamma and no seeds")
        elif self.algorithm in {"q_learning", "sarsa_lambda"}:
            if self.reward_mode not in {"sparse", "shaped"} or self.schedule not in {"linear", "exponential", "geometric"}:
                raise ValueError("model-free slots require reward_mode and schedule")
            if not self.seeds or self.episodes is None or self.episodes <= 0:
                raise ValueError("model-free slots require seeds and positive episodes")
            if self.algorithm == "sarsa_lambda" and (self.trace_lambda is None or not 0 <= self.trace_lambda <= 1):
                raise ValueError("SARSA slots require lambda in [0,1]")
        else:
            if self.available or not self.unavailable_reason or not self.transfer_role or not self.transfer_variant or not self.seeds:
                raise ValueError("transfer slots must be visible and unavailable")


@dataclass(frozen=True, slots=True)
class Preset:
    name: str
    source: Path
    project_config: Path
    output_root: Path
    final: bool
    evaluation: EvaluationSpec
    slots: tuple[ScientificSlot, ...]
    event_audit: bool = True

    def __post_init__(self) -> None:
        if self.name not in BUILTIN_PRESETS:
            raise ValueError(f"unknown preset {self.name!r}")
        keys = [slot.key for slot in self.slots]
        if len(keys) != len(set(keys)):
            raise ValueError("preset contains duplicate slot keys")
        if self.final != (self.name in FINAL_PRESETS):
            raise ValueError("preset final flag disagrees with its name")


@dataclass(frozen=True, slots=True)
class ResolvedRun:
    preset: str
    slot_key: str
    slot_id: str
    config_hash: str
    algorithm: str
    root_seed: int | None
    seeds: SeedStreams | None
    parameters: Mapping[str, Any]
    available: bool
    unavailable_reason: str | None


@dataclass(frozen=True, slots=True)
class EvaluationEpisode:
    episode: int
    tie_seed: int
    transition_seed: int
    steps: int
    base_return: float
    shaping_return: float
    total_return: float
    learning_return: float | None
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


@dataclass(frozen=True, slots=True)
class EvaluationStep:
    episode: int
    step: int
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
    terminated: bool
    truncated: bool


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    episodes: tuple[EvaluationEpisode, ...]
    steps: tuple[EvaluationStep, ...]
    tie_seeds: tuple[int, ...]
    transition_seeds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EventWitness:
    event: str
    source_key: int
    source_row: int
    source_col: int
    intended_action: str
    actual_action: str
    outcome_probability: float
    next_key: int
    next_row: int
    next_col: int
    co_events: str
    source: str


@dataclass(frozen=True, slots=True)
class ResumeDecision:
    skip: bool
    reason: str
    attempt_path: Path | None = None


@dataclass(frozen=True, slots=True)
class RunSummary:
    preset: str
    complete: int
    skipped: int
    failed: int
    unavailable: int
    selected: int
    output_directory: Path


def _builtin_slots(name: str) -> tuple[ScientificSlot, ...]:
    vi = tuple(ScientificSlot(f"vi-{mode}-g{gamma:.2f}", "vi", mode, gamma)
               for mode, gamma in (("shaped", .90), ("shaped", .95), ("shaped", .99), ("sparse", .95)))
    final_seeds = (9, 109, 209, 309, 409)
    smoke_seeds = (9,)
    q_final = (
        ScientificSlot("q-shaped-linear", "q_learning", "shaped", .95, "linear", episodes=5000, seeds=final_seeds),
        ScientificSlot("q-shaped-exponential", "q_learning", "shaped", .95, "exponential", episodes=5000, seeds=final_seeds),
        ScientificSlot("q-sparse-linear", "q_learning", "sparse", .95, "linear", episodes=5000, seeds=final_seeds),
    )
    q_smoke = tuple(ScientificSlot(s.key, s.algorithm, s.reward_mode, s.gamma, s.schedule,
                                  episodes=40, seeds=smoke_seeds,
                                  audit_episode=1 if s.key == "q-shaped-linear" else None)
                    for s in q_final)
    s_final = tuple(ScientificSlot(f"sarsa-shaped-linear-l{lam:.1f}", "sarsa_lambda", "shaped", .95,
                                   "linear", lam, 5000, final_seeds)
                    for lam in (0., .3, .7, .9))
    s_smoke = tuple(ScientificSlot(s.key, s.algorithm, s.reward_mode, s.gamma, s.schedule,
                                  s.trace_lambda, 40, smoke_seeds,
                                  diagnostic_episode=1 if s.trace_lambda == 0 else None)
                    for s in s_final)
    transfer = tuple(ScientificSlot(f"transfer-{role}-{variant}", "transfer", seeds=final_seeds,
                                    transfer_role=role, transfer_variant=variant, available=False,
                                    unavailable_reason=TRANSFER_UNAVAILABLE_REASON)
                     for role in ("target-a", "target-b")
                     for variant in ("scratch", "full", "scaled", "selective"))
    if name == "smoke":
        return vi + q_smoke + s_smoke + tuple(ScientificSlot(
            s.key, s.algorithm, seeds=smoke_seeds, transfer_role=s.transfer_role,
            transfer_variant=s.transfer_variant, available=False,
            unavailable_reason=s.unavailable_reason) for s in transfer)
    if name == "final-vi": return vi
    if name == "final-q": return q_final
    if name == "final-sarsa": return s_final
    if name == "final-transfer": return transfer
    if name == "final-all": return vi + q_final + s_final + transfer
    raise ValueError(f"unknown preset {name!r}")


def _builtin_preset(name: str, *, output_root: Path | None = None) -> Preset:
    source = DEFAULT_PRESET_DIR / f"{name}.yaml"
    return Preset(name, source.resolve(), DEFAULT_CONFIG_PATH.resolve(),
                  (output_root or DEFAULT_OUTPUT_ROOT).resolve(), name in FINAL_PRESETS,
                  EvaluationSpec(3 if name == "smoke" else 100, True), _builtin_slots(name))


def _parse_slot(raw: Any, index: int) -> ScientificSlot:
    allowed = {"key", "algorithm", "reward_mode", "gamma", "schedule", "lambda", "episodes", "seeds",
               "audit_episode", "diagnostic_episode", "transfer_role", "transfer_variant", "available", "unavailable_reason"}
    item = _strict_keys(raw, f"slots[{index}]", allowed, {"key", "algorithm"})
    seeds_raw = item.get("seeds", [])
    if not isinstance(seeds_raw, list):
        raise ValueError(f"slots[{index}].seeds must be a list")
    seeds = tuple(_integer(x, f"slots[{index}].seeds") for x in seeds_raw)
    return ScientificSlot(
        str(item["key"]), str(item["algorithm"]),
        None if item.get("reward_mode") is None else str(item["reward_mode"]),
        None if item.get("gamma") is None else _number(item["gamma"], "slot.gamma"),
        None if item.get("schedule") is None else str(item["schedule"]),
        None if item.get("lambda") is None else _number(item["lambda"], "slot.lambda"),
        None if item.get("episodes") is None else _integer(item["episodes"], "slot.episodes", 1), seeds,
        None if item.get("audit_episode") is None else _integer(item["audit_episode"], "slot.audit_episode", 1),
        None if item.get("diagnostic_episode") is None else _integer(item["diagnostic_episode"], "slot.diagnostic_episode", 1),
        item.get("transfer_role"), item.get("transfer_variant"), bool(item.get("available", True)),
        item.get("unavailable_reason"))


def load_preset(name_or_path: str | Path, *, output_root: str | Path | None = None) -> Preset:
    """Load one strict YAML preset, or synthesize a locked built-in if absent."""
    candidate = Path(name_or_path)
    if candidate.suffix.lower() not in {".yaml", ".yml"} and str(name_or_path) in BUILTIN_PRESETS:
        candidate = DEFAULT_PRESET_DIR / f"{name_or_path}.yaml"
    if not candidate.exists():
        name = candidate.stem
        if name not in BUILTIN_PRESETS:
            raise ValueError(f"preset does not exist: {candidate}")
        return _builtin_preset(name, output_root=None if output_root is None else Path(output_root))
    source = candidate.resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"could not read preset {source}: {exc}") from exc
    root = _strict_keys(raw, "preset root", {"schema_version", "preset", "execution", "seeds", "evaluation", "audits", "matrix", "expected_counts", "confirmation_metadata"},
                        {"schema_version", "preset", "execution", "seeds", "evaluation", "audits", "matrix", "expected_counts", "confirmation_metadata"})
    if root["schema_version"] != PRESET_SCHEMA_VERSION:
        raise ValueError("unsupported preset schema_version")
    meta = _strict_keys(root["preset"], "preset", {"name", "kind", "description", "project_config", "output_subdir"},
                        {"name", "kind", "description", "project_config", "output_subdir"})
    execution = _strict_keys(root["execution"], "execution", {"require_confirmation", "confirmation_flag", "sequential", "resume"},
                             {"require_confirmation", "confirmation_flag", "sequential", "resume"})
    seeds_doc = _strict_keys(root["seeds"], "seeds", {"roots", "stream_derivation", "stream_order"},
                             {"roots", "stream_derivation", "stream_order"})
    roots = seeds_doc["roots"]
    if not isinstance(roots, list): raise ValueError("seeds.roots must be a list")
    roots = tuple(_integer(x, "seeds.roots") for x in roots)
    if seeds_doc["stream_derivation"] != "numpy-seedsequence-spawn-v1": raise ValueError("unsupported seed derivation")
    expected_order = ["training_action_choice", "training_transitions", "evaluation_tie_breaking", "evaluation_transitions", "statistics"]
    if seeds_doc["stream_order"] != expected_order: raise ValueError("seed stream order is not the locked five-stream order")
    evaluation = _strict_keys(root["evaluation"], "evaluation", {"policy", "common_rollout_group", "rollouts", "max_steps", "preserve_arrays", "record_trajectories", "transfer_checkpoints"},
                              {"policy", "common_rollout_group", "rollouts", "max_steps", "preserve_arrays", "record_trajectories"})
    if evaluation["policy"] != "greedy_exact_max_uniform_ties" or evaluation["preserve_arrays"] is not True: raise ValueError("unsupported evaluation semantics")
    audits = _strict_keys(root["audits"], "audits", {"event_coverage", "q_step_audits", "sarsa_trace_diagnostics"},
                          {"event_coverage", "q_step_audits", "sarsa_trace_diagnostics"})
    matrix = _strict_keys(root["matrix"], "matrix", {"value_iteration", "q_learning", "sarsa_lambda", "transfer_q_learning"},
                          {"value_iteration", "q_learning", "sarsa_lambda", "transfer_q_learning"})
    q_audits = {str(x["slot"]): _integer(x["episode"], "q audit episode", 1) for x in audits["q_step_audits"]}
    s_audits = {str(x["slot"]): _integer(x["episode"], "SARSA diagnostic episode", 1) for x in audits["sarsa_trace_diagnostics"]}
    slots: list[ScientificSlot] = []
    vi = _strict_keys(matrix["value_iteration"], "matrix.value_iteration", {"defaults", "variants"}, {"defaults", "variants"})
    for variant in vi["variants"]:
        item = _strict_keys(variant, "VI variant", {"slot", "reward_mode", "gamma"}, {"slot", "reward_mode", "gamma"})
        slots.append(ScientificSlot(str(item["slot"]), "vi", str(item["reward_mode"]), _number(item["gamma"], "VI gamma")))
    q = _strict_keys(matrix["q_learning"], "matrix.q_learning", {"defaults", "variants"}, {"defaults", "variants"})
    q_defaults = q["defaults"]
    if not isinstance(q_defaults, Mapping): raise ValueError("Q defaults must be a mapping")
    for variant in q["variants"]:
        item = {**q_defaults, **dict(variant)}; key = str(item["slot"])
        slots.append(ScientificSlot(key, "q_learning", str(item["reward_mode"]), _number(item["gamma"], "Q gamma"),
            str(item["schedule"]), episodes=_integer(item["episodes"], "Q episodes", 1), seeds=roots,
            audit_episode=q_audits.get(key, item.get("audit_episode"))))
    sarsa = _strict_keys(matrix["sarsa_lambda"], "matrix.sarsa_lambda", {"defaults", "variants"}, {"defaults", "variants"})
    s_defaults = sarsa["defaults"]
    if not isinstance(s_defaults, Mapping): raise ValueError("SARSA defaults must be a mapping")
    for variant in sarsa["variants"]:
        item = {**s_defaults, **dict(variant)}; key = str(item["slot"])
        slots.append(ScientificSlot(key, "sarsa_lambda", str(item["reward_mode"]), _number(item["gamma"], "SARSA gamma"),
            str(item["schedule"]), _number(item["lambda"], "SARSA lambda"), _integer(item["episodes"], "SARSA episodes", 1), roots,
            diagnostic_episode=s_audits.get(key, item.get("diagnostic_episode"))))
    transfer = _strict_keys(matrix["transfer_q_learning"], "matrix.transfer_q_learning", {"defaults", "target_roles", "initializations"},
                            {"defaults", "target_roles", "initializations"})
    t_defaults = transfer["defaults"]
    if not isinstance(t_defaults, Mapping): raise ValueError("transfer defaults must be a mapping")
    for role in transfer["target_roles"]:
        role_doc = _strict_keys(role, "transfer role", {"role", "map"}, {"role", "map"})
        _resolved_path(role_doc["map"], "transfer map", source.parent)
        for initialization in transfer["initializations"]:
            init = _strict_keys(initialization, "transfer initialization", {"name", "beta", "signature"}, {"name"})
            reason = str(t_defaults.get("unavailable_reason", ""))
            if t_defaults.get("status") != "unavailable" or not reason: raise ValueError("Phase 5 transfer slots must be unavailable")
            key = f"transfer-{role_doc['role']}-{init['name']}"
            slots.append(ScientificSlot(key, "transfer", seeds=roots, transfer_role=str(role_doc["role"]),
                transfer_variant=str(init["name"]), available=False, unavailable_reason=reason))
    name = str(meta["name"]); final = meta["kind"] == "final"
    if bool(execution["require_confirmation"]) != final: raise ValueError("confirmation setting disagrees with preset kind")
    expected = _strict_keys(root["expected_counts"], "expected_counts", {"vi_solves", "q_runs", "sarsa_runs", "transfer_slots", "ready_slots", "unavailable_slots", "total_slots", "model_free_slots"},
                            {"vi_solves", "q_runs", "sarsa_runs", "transfer_slots", "ready_slots", "unavailable_slots", "total_slots"})
    preset = Preset(name, source, _resolved_path(meta["project_config"], "project_config", source.parent),
        Path(output_root).resolve() if output_root is not None else DEFAULT_OUTPUT_ROOT.resolve(), final,
        EvaluationSpec(_integer(evaluation["rollouts"], "evaluation.rollouts", 1), bool(evaluation["record_trajectories"]), str(evaluation["common_rollout_group"])),
        tuple(slots), bool(audits["event_coverage"]))
    expanded = expand_preset(preset)
    observed = {"vi_solves":sum(r.algorithm=="vi" for r in expanded), "q_runs":sum(r.algorithm=="q_learning" for r in expanded),
        "sarsa_runs":sum(r.algorithm=="sarsa_lambda" for r in expanded), "transfer_slots":sum(r.algorithm=="transfer" for r in expanded),
        "ready_slots":sum(r.available for r in expanded), "unavailable_slots":sum(not r.available for r in expanded), "total_slots":len(expanded)}
    for key,value in observed.items():
        if _integer(expected[key], f"expected_counts.{key}") != value: raise ValueError(f"expected count mismatch for {key}: {value}")
    return preset


def list_presets() -> tuple[str, ...]:
    return BUILTIN_PRESETS


def expand_preset(preset: Preset) -> tuple[ResolvedRun, ...]:
    runs: list[ResolvedRun] = []
    for slot in preset.slots:
        roots: tuple[int | None, ...] = slot.seeds if slot.seeds else (None,)
        for root in roots:
            parameters = {key: value for key, value in asdict(slot).items()
                          if key not in {"key", "seeds", "available", "unavailable_reason"} and value is not None}
            identity = {"schema_version": PRESET_SCHEMA_VERSION, "preset": preset.name,
                        "slot_key": slot.key, "root_seed": root, "parameters": parameters,
                        "evaluation": asdict(preset.evaluation)}
            digest = stable_hash(identity)
            runs.append(ResolvedRun(preset.name, slot.key, digest[:16], digest, slot.algorithm, root,
                                    None if root is None else derive_seed_streams(root), parameters,
                                    slot.available, slot.unavailable_reason))
    if len({(run.slot_key, run.root_seed) for run in runs}) != len(runs):
        raise ValueError("preset expansion produced duplicate concrete slots")
    return tuple(runs)


def common_rollout_seeds(streams: SeedStreams | None, evaluation: EvaluationSpec, *, fallback_root: int = 0) -> tuple[tuple[int, ...], tuple[int, ...]]:
    tie_root = streams.evaluation_tie_breaking if streams else derive_seed_streams(fallback_root).evaluation_tie_breaking
    transition_root = streams.evaluation_transition if streams else derive_seed_streams(fallback_root).evaluation_transition
    group = int.from_bytes(hashlib.sha256(evaluation.group.encode()).digest()[:8], "big")
    tie = np.random.SeedSequence([tie_root, group]).spawn(evaluation.episodes)
    transition = np.random.SeedSequence([transition_root, group]).spawn(evaluation.episodes)
    convert = lambda xs: tuple(int(x.generate_state(1, dtype=np.uint64)[0]) for x in xs)
    return convert(tie), convert(transition)


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def evaluate_q_values(mdp: MazeMDP, q_values: np.ndarray, evaluation: EvaluationSpec, *,
                      streams: SeedStreams | None = None, fallback_root: int = 0,
                      learning_reward_mode: str | None = None) -> EvaluationResult:
    """Evaluate an exact-max policy with uniform ties and no learning/mutation."""
    expected = (2, mdp.spec.rows, mdp.spec.cols, len(ACTION_ORDER))
    if q_values.shape != expected:
        raise ValueError(f"q_values must have shape {expected}")
    before = _array_digest(q_values)
    tie_seeds, transition_seeds = common_rollout_seeds(streams, evaluation, fallback_root=fallback_root)
    episode_rows, step_rows = [], []
    action_index = {action: i for i, action in enumerate(ACTION_ORDER)}
    for number, (tie_seed, transition_seed) in enumerate(zip(tie_seeds, transition_seeds, strict=True), 1):
        started = time.perf_counter(); tie_rng = np.random.default_rng(tie_seed)
        env = MazeEpisode(mdp, seed=transition_seed); state = env.reset(); visited = {state}
        events: Counter[EventType] = Counter(); base = shaping = 0.0
        while not env.done:
            source = state; values = q_values[state_index(source)]
            if not np.all(np.isfinite(values)):
                raise ValueError("evaluation reached non-finite Q values")
            maxima = np.flatnonzero(values == np.max(values))
            chosen_index = int(tie_rng.choice(maxima)); chosen = ACTION_ORDER[chosen_index]
            result = env.step(chosen); state = result.state; visited.add(state)
            base += result.base_reward; shaping += result.shaping_reward; events.update(result.events)
            if evaluation.record_steps:
                step_rows.append(EvaluationStep(number, result.step_number, int(source.has_key), source.row,
                    source.col, chosen.name, chosen_index, result.actual_action.name,
                    action_index[result.actual_action], result.probability, int(state.has_key), state.row,
                    state.col, "|".join(event.value for event in result.events), result.base_reward,
                    result.shaping_reward, result.total_reward, result.terminated, result.truncated))
        total = base + shaping
        learning = None if learning_reward_mode is None else (base if learning_reward_mode == "sparse" else total)
        episode_rows.append(EvaluationEpisode(number, tie_seed, transition_seed, env.elapsed_steps,
            base, shaping, total, learning, bool(events[EventType.GOAL_REACHED]),
            bool(events[EventType.GOAL_REACHED]), bool(events[EventType.EPISODE_TRUNCATED]),
            int(events[EventType.MOVE]), int(events[EventType.WALL_COLLISION]),
            int(events[EventType.PENALTY_ENTERED]), int(events[EventType.KEY_COLLECTED]),
            int(events[EventType.CLOSED_DOOR_ATTEMPT]), int(events[EventType.DOOR_PASSED]),
            int(events[EventType.TELEPORTED]), int(events[EventType.GOAL_REACHED]),
            int(events[EventType.EPISODE_TRUNCATED]), len(visited), env.elapsed_steps + 1 - len(visited),
            time.perf_counter() - started))
    if _array_digest(q_values) != before:
        raise RuntimeError("evaluation mutated the Q array")
    return EvaluationResult(tuple(episode_rows), tuple(step_rows), tie_seeds, transition_seeds)


def audit_event_coverage(mdp: MazeMDP) -> tuple[EventWitness, ...]:
    """Find transition-model witnesses for every stable event plus real truncation."""
    witnesses: dict[EventType, EventWitness] = {}
    queue = deque([mdp.initial_state()]); seen = {mdp.initial_state()}
    while queue:
        state = queue.popleft()
        if state.position == mdp.spec.goal:
            continue
        for action in ACTION_ORDER:
            for outcome in mdp.transition_outcomes(state, action):
                for event in outcome.events:
                    witnesses.setdefault(event, EventWitness(event.value, int(state.has_key), state.row,
                        state.col, action.name, outcome.actual_action.name, outcome.probability,
                        int(outcome.state.has_key), outcome.state.row, outcome.state.col,
                        "|".join(x.value for x in outcome.events), "transition_outcomes"))
                if outcome.probability > 0 and outcome.state not in seen:
                    seen.add(outcome.state); queue.append(outcome.state)
    truncation = MazeEpisode(mdp, seed=0, max_steps=1)
    source = truncation.reset(); step = truncation.step(Action.UP)
    if not step.truncated:
        for action in ACTION_ORDER:
            truncation.reset(seed=0); source = truncation.state; step = truncation.step(action)
            if step.truncated: break
    if step.truncated:
        witnesses[EventType.EPISODE_TRUNCATED] = EventWitness(EventType.EPISODE_TRUNCATED.value,
            int(source.has_key), source.row, source.col, step.intended_action.name,
            step.actual_action.name, step.probability, int(step.state.has_key), step.state.row,
            step.state.col, "|".join(x.value for x in step.events), "MazeEpisode(max_steps=1)")
    missing = [event.value for event in EventType if event not in witnesses]
    if missing:
        raise ValueError(f"event coverage is incomplete: {missing}")
    return tuple(witnesses[event] for event in EventType)


def _git_provenance(root: Path, *, excluded_paths: Sequence[Path] = ()) -> dict[str, Any]:
    def command(*args: str) -> str | None:
        try:
            result = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                                    text=True, timeout=10, check=False)
            return result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None
    commit = command("rev-parse", "HEAD")
    status = command("status", "--porcelain=v1", "--untracked-files=all")
    if status is not None and excluded_paths:
        prefixes: list[str] = []
        for path in excluded_paths:
            try:
                prefixes.append(path.resolve().relative_to(root.resolve()).as_posix().rstrip("/") + "/")
            except ValueError:
                continue
        status = "\n".join(
            line for line in status.splitlines()
            if not any(line[3:].replace("\\", "/").lstrip('"').startswith(prefix) for prefix in prefixes)
        )
    return {"commit": commit, "dirty": bool(status) if status is not None else None,
            "dirty_fingerprint": None if status is None else hashlib.sha256(status.encode()).hexdigest()}


def repository_provenance(*, excluded_paths: Sequence[Path] = ()) -> dict[str, Any]:
    versions = {}
    for name in ("numpy", "PyYAML"):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    return {"git": _git_provenance(PROJECT_ROOT, excluded_paths=excluded_paths), "python": platform.python_version(),
            "implementation": platform.python_implementation(), "platform": platform.platform(),
            "executable": sys.executable, "dependencies": versions}


def _validate_project(preset: Preset) -> tuple[OperationalConfig, MazeSpec]:
    config = load_config(preset.project_config)
    spec = load_source_map(config.source_map)
    if spec.student_id != config.student_id or spec.base_seed != config.base_seed:
        raise ValueError("project config identity does not match source map")
    return config, spec


def _run_document(preset: Preset, run: ResolvedRun, config: OperationalConfig, spec: MazeSpec) -> dict[str, Any]:
    return {"schema_version": PRESET_SCHEMA_VERSION, "preset": preset.name, "slot_key": run.slot_key,
            "slot_id": run.slot_id, "config_hash": run.config_hash, "algorithm": run.algorithm,
            "parameters": dict(run.parameters), "root_seed": run.root_seed,
            "seed_streams": None if run.seeds is None else asdict(run.seeds),
            "evaluation": asdict(preset.evaluation), "project_config": str(config.path),
            "source_map": str(config.source_map), "map_checksum": map_checksum(spec),
            "map_document_hash": stable_hash(source_map_document(spec))}


def _csv_value(value: Any) -> Any:
    if value is None: return ""
    if isinstance(value, bool): return int(value)
    if isinstance(value, float): return repr(value)
    return value


def _write_csv(path: Path, rows: Sequence[Any], provenance: Mapping[str, Any], row_fields: Sequence[str] | None = None) -> int:
    if row_fields is None:
        row_fields = [field.name for field in fields(type(rows[0]))] if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*provenance, *row_fields])
        writer.writeheader()
        for row in rows:
            values = asdict(row) if is_dataclass(row) else dict(row)
            writer.writerow({**{k: _csv_value(v) for k, v in provenance.items()},
                             **{k: _csv_value(values.get(k)) for k in row_fields}})
    return len(rows)


def _normalized_episode_rows(training: Sequence[Any], evaluation: EvaluationResult) -> list[dict[str, Any]]:
    names = [f.name for f in fields(EvaluationEpisode)]
    result = []
    for item in training:
        raw = asdict(item)
        result.append({"phase": "training", **{name: raw.get(name) for name in names},
                       "tie_seed": None, "transition_seed": None})
    result.extend({"phase": "evaluation", **asdict(item)} for item in evaluation.episodes)
    return result


def _artifact(path: Path, row_count: int | None = None) -> dict[str, Any]:
    return {"path": path.name, "sha256": _sha256_file(path), "bytes": path.stat().st_size,
            **({} if row_count is None else {"row_count": row_count})}


def _strict_model(path: Path, algorithm: str, spec: MazeSpec) -> None:
    if algorithm == "vi": load_value_iteration_npz(path, expected_spec=spec)
    elif algorithm == "q_learning": load_q_learning_npz(path, expected_spec=spec)
    elif algorithm == "sarsa_lambda": load_sarsa_lambda_npz(path, expected_spec=spec)
    else: raise ValueError("unknown model algorithm")


def validate_attempt(path: str | Path, *, expected: Mapping[str, Any] | None = None,
                     spec: MazeSpec | None = None, provenance: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    directory = Path(path).resolve(); ledger_path = directory / "attempt.json"
    try: ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc: raise ValueError(f"invalid attempt ledger: {exc}") from exc
    if ledger.get("schema_version") != ATTEMPT_SCHEMA_VERSION or ledger.get("status") != "complete":
        raise ValueError("attempt is not complete")
    if expected:
        for key in ("config_hash", "map_checksum", "map_document_hash"):
            if ledger.get(key) != expected.get(key): raise ValueError(f"attempt {key} mismatch")
    if provenance:
        old, current = ledger.get("provenance", {}).get("git", {}), provenance.get("git", {})
        for key in ("commit", "dirty_fingerprint"):
            if old.get(key) != current.get(key): raise ValueError(f"attempt Git {key} mismatch")
    artifacts = ledger.get("artifacts")
    if not isinstance(artifacts, Mapping): raise ValueError("attempt artifacts are invalid")
    for name, entry in artifacts.items():
        artifact_path = directory / entry["path"]
        if not artifact_path.is_file() or artifact_path.stat().st_size != entry["bytes"] or _sha256_file(artifact_path) != entry["sha256"]:
            raise ValueError(f"artifact validation failed: {name}")
        if "row_count" in entry:
            with artifact_path.open(encoding="utf-8", newline="") as handle:
                count = sum(1 for _ in csv.DictReader(handle))
            if count != entry["row_count"]: raise ValueError(f"artifact row count failed: {name}")
    if spec is not None: _strict_model(directory / artifacts["model"]["path"], ledger["algorithm"], spec)
    return ledger


def _attempts(slot_directory: Path) -> list[Path]:
    return sorted((p for p in slot_directory.glob("attempt-*") if p.is_dir()), key=lambda p: p.name)


def resume_decision(slot_directory: Path, expected: Mapping[str, Any], spec: MazeSpec,
                    provenance: Mapping[str, Any]) -> ResumeDecision:
    for attempt in reversed(_attempts(slot_directory)):
        try:
            validate_attempt(attempt, expected=expected, spec=spec, provenance=provenance)
            return ResumeDecision(True, "matching validated complete attempt", attempt)
        except ValueError:
            continue
    return ResumeDecision(False, "no matching validated complete attempt")


def _execute_attempt(preset: Preset, run: ResolvedRun, config: OperationalConfig, spec: MazeSpec,
                     provenance: Mapping[str, Any], *, forced_failure_stage: str | None = None) -> Path:
    slot_dir = preset.output_root / preset.name / "attempts" / run.slot_id
    number = len(_attempts(slot_dir)) + 1
    attempt_dir = slot_dir / f"attempt-{number:04d}-{uuid.uuid4().hex[:12]}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    resolved = _run_document(preset, run, config, spec)
    _atomic_json(attempt_dir / "resolved_config.json", resolved)
    _atomic_json(attempt_dir / "map.json", source_map_document(spec))
    ledger: dict[str, Any] = {"schema_version": ATTEMPT_SCHEMA_VERSION, "status": "running",
        "stage": "initializing", "started_at": _utc_now(), "preset": preset.name,
        "slot_key": run.slot_key, "slot_id": run.slot_id, "algorithm": run.algorithm,
        "config_hash": run.config_hash, "map_checksum": resolved["map_checksum"],
        "map_document_hash": resolved["map_document_hash"], "root_seed": run.root_seed,
        "seed_streams": resolved["seed_streams"], "provenance": provenance, "artifacts": {}}
    _atomic_json(attempt_dir / "attempt.json", ledger)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    # Full-run tracemalloc makes the allocation-heavy VI sweeps several times slower.
    # Track Python allocations for model-free attempts; VI records zero as not sampled.
    memory_traced = run.algorithm != "vi"
    if memory_traced:
        tracemalloc.start()
    try:
        if forced_failure_stage == "training": raise RuntimeError("controlled training failure")
        reward_mode = run.parameters.get("reward_mode", "shaped")
        gamma = float(run.parameters.get("gamma", .95))
        mdp = MazeMDP(spec, config.rewards, gamma=gamma, use_shaping=reward_mode == "shaped")
        training_rows: Sequence[Any] = (); diagnostic_rows: Sequence[Any] = ()
        if run.algorithm == "vi":
            vi_config = ValueIterationConfig(gamma, reward_mode, config.planning.theta,
                                             config.planning.max_sweeps, config.planning.tie_tolerance)
            result = value_iteration(mdp, vi_config); model = attempt_dir / "model.npz"
            save_value_iteration_npz(model, values=result.values, q_values=result.q_values,
                optimal_action_mask=result.optimal_action_mask, valid_mask=result.valid_state_mask,
                reachable_mask=result.reachable_state_mask, terminal_mask=result.terminal_state_mask,
                delta_history=result.delta_history, metadata=result.metadata())
            q_values = result.q_values; learner_id = None
        elif run.algorithm == "q_learning":
            qcfg = QLearningConfig(config.q_learning.gamma, config.q_learning.alpha,
                int(run.parameters["episodes"]), config.q_learning.epsilon_start,
                config.q_learning.epsilon_end, min(config.q_learning.decay_episodes, int(run.parameters["episodes"])),
                str(run.parameters["schedule"]), reward_mode, run.parameters.get("audit_episode"),
                config.q_learning.shaping_method, config.q_learning.shaping_version)
            assert run.seeds is not None
            # The current strict model loader accepts only the historical derivation label;
            # the complete five-stream record remains authoritative in resolved_config/attempt.
            learner_seeds = QLearningSeeds(run.seeds.root, run.seeds.training_action,
                                           run.seeds.training_transition)
            result = train_q_learning(mdp, qcfg, root_seed=run.root_seed, learner_seeds=learner_seeds)
            model = attempt_dir / "model.npz"
            save_q_learning_npz(model, q_values=result.q_values, state_visit_counts=result.state_visit_counts,
                state_action_visit_counts=result.state_action_visit_counts, valid_mask=result.valid_state_mask,
                reachable_mask=result.reachable_state_mask, terminal_mask=result.terminal_state_mask,
                metadata=result.metadata())
            q_values=result.q_values; training_rows=result.episode_metrics; diagnostic_rows=result.audit_rows
            learner_id=result.identity.run_id
        elif run.algorithm == "sarsa_lambda":
            scfg = SarsaLambdaConfig(config.sarsa_lambda.gamma, config.sarsa_lambda.alpha,
                float(run.parameters["trace_lambda"]), int(run.parameters["episodes"]),
                config.sarsa_lambda.epsilon_start, config.sarsa_lambda.epsilon_end,
                min(config.sarsa_lambda.decay_episodes, int(run.parameters["episodes"])),
                str(run.parameters["schedule"]), reward_mode, run.parameters.get("diagnostic_episode"),
                config.sarsa_lambda.shaping_method, config.sarsa_lambda.shaping_version)
            assert run.seeds is not None
            learner_seeds = SarsaLambdaSeeds(run.seeds.root, run.seeds.training_action,
                                              run.seeds.training_transition)
            result = train_sarsa_lambda(mdp, scfg, root_seed=run.root_seed, learner_seeds=learner_seeds)
            model = attempt_dir / "model.npz"
            save_sarsa_lambda_npz(model, q_values=result.q_values, state_visit_counts=result.state_visit_counts,
                state_action_visit_counts=result.state_action_visit_counts, valid_mask=result.valid_state_mask,
                reachable_mask=result.reachable_state_mask, terminal_mask=result.terminal_state_mask,
                metadata=result.metadata())
            q_values=result.q_values; training_rows=result.episode_metrics; diagnostic_rows=result.diagnostic_rows
            learner_id=result.identity.run_id
        else: raise RuntimeError("unavailable transfer slot cannot execute")
        ledger["stage"] = "evaluation"; _atomic_json(attempt_dir / "attempt.json", ledger)
        evaluation = evaluate_q_values(mdp, q_values, preset.evaluation, streams=run.seeds,
                                       fallback_root=config.base_seed, learning_reward_mode=reward_mode)
        provenance_row = {"csv_schema_version": CSV_SCHEMA_VERSION, "preset": preset.name,
            "slot_key": run.slot_key, "slot_id": run.slot_id, "attempt": attempt_dir.name,
            "algorithm": run.algorithm, "config_hash": run.config_hash, "learner_id": learner_id or "",
            "map_checksum": resolved["map_checksum"], "root_seed": run.root_seed}
        episode_fields = ["phase", *[f.name for f in fields(EvaluationEpisode)]]
        episode_rows = _normalized_episode_rows(training_rows, evaluation)
        episodes_path = attempt_dir / "episodes.csv"
        episode_count = _write_csv(episodes_path, episode_rows, provenance_row, episode_fields)
        artifacts = {"model": _artifact(model), "episodes": _artifact(episodes_path, episode_count)}
        if diagnostic_rows or evaluation.steps:
            rows = []
            if diagnostic_rows:
                rows.extend({"phase": "training", **asdict(row)} for row in diagnostic_rows)
            rows.extend({"phase": "evaluation", **asdict(row)} for row in evaluation.steps)
            keys = sorted(set().union(*(row.keys() for row in rows))) if rows else ["phase"]
            steps_path = attempt_dir / "steps.csv"
            step_count = _write_csv(steps_path, rows, provenance_row, keys)
            artifacts["steps"] = _artifact(steps_path, step_count)
        if run.algorithm == "sarsa_lambda" and diagnostic_rows:
            trace_path = attempt_dir / "trace_diagnostic.csv"
            trace_count = _write_csv(trace_path, diagnostic_rows, provenance_row)
            artifacts["trace_diagnostic"] = _artifact(trace_path, trace_count)
        _strict_model(model, run.algorithm, spec)
        if forced_failure_stage == "validation": raise RuntimeError("controlled validation failure")
        current, peak = tracemalloc.get_traced_memory() if memory_traced else (0, 0)
        ledger.update({"status": "complete", "stage": "complete", "completed_at": _utc_now(),
            "wall_runtime_seconds": time.perf_counter()-wall_start,
            "cpu_runtime_seconds": time.process_time()-cpu_start, "peak_python_bytes": peak,
            "learner_id": learner_id, "artifacts": artifacts,
            "evaluation_seed_vectors": {"tie_breaking": evaluation.tie_seeds,
                                        "transitions": evaluation.transition_seeds}})
        _atomic_json(attempt_dir / "attempt.json", ledger)
        validate_attempt(attempt_dir, expected=resolved, spec=spec, provenance=provenance)
        return attempt_dir
    except BaseException as exc:
        current, peak = tracemalloc.get_traced_memory() if memory_traced else (0, 0)
        ledger.update({"status": "failed", "completed_at": _utc_now(),
            "wall_runtime_seconds": time.perf_counter()-wall_start,
            "cpu_runtime_seconds": time.process_time()-cpu_start, "peak_python_bytes": peak,
            "failure": {"type": type(exc).__name__, "message": str(exc),
                        "traceback": traceback.format_exc()}})
        _atomic_json(attempt_dir / "attempt.json", ledger)
        raise
    finally:
        if memory_traced:
            tracemalloc.stop()


def _latest_slot_state(preset: Preset, run: ResolvedRun) -> dict[str, Any]:
    if not run.available:
        return {"status": "unavailable", "latest_failure": run.unavailable_reason,
                "attempt_count": 0, "latest_attempt": ""}
    slot_dir = preset.output_root / preset.name / "attempts" / run.slot_id
    attempts = _attempts(slot_dir)
    if not attempts: return {"status": "missing", "latest_failure": "", "attempt_count": 0, "latest_attempt": ""}
    latest, last_failure = attempts[-1], ""
    try: ledger = json.loads((latest / "attempt.json").read_text(encoding="utf-8"))
    except Exception: return {"status": "running/incomplete", "latest_failure": "invalid ledger", "attempt_count": len(attempts), "latest_attempt": latest.name}
    status = ledger.get("status")
    if status == "failed": last_failure = ledger.get("failure", {}).get("message", "")
    if status not in {"complete", "failed"}: status = "running/incomplete"
    return {"status": status, "latest_failure": last_failure, "attempt_count": len(attempts), "latest_attempt": latest.name}


def refresh_indexes(preset: Preset, runs: Sequence[ResolvedRun], *, event_witnesses: Sequence[EventWitness] = ()) -> Mapping[str, Any]:
    directory = preset.output_root / preset.name; rows=[]
    for run in runs:
        state = _latest_slot_state(preset, run)
        rows.append({"preset":preset.name,"slot_key":run.slot_key,"slot_id":run.slot_id,
                     "algorithm":run.algorithm,"root_seed":run.root_seed,"config_hash":run.config_hash,**state})
    slots_path = directory / "slots.csv"
    temp = slots_path.with_name(f".{slots_path.name}.{uuid.uuid4().hex}.tmp")
    _write_csv(temp, rows, {}, list(rows[0]) if rows else [])
    slots_path.parent.mkdir(parents=True, exist_ok=True); os.replace(temp, slots_path)
    counts = Counter(row["status"] for row in rows)
    manifest = {"schema_version": MANIFEST_SCHEMA_VERSION, "preset": preset.name,
        "generated_at": _utc_now(), "planned": len(rows), "ready": sum(run.available for run in runs),
        "complete": counts["complete"], "failed": counts["failed"], "missing": counts["missing"],
        "running_incomplete": counts["running/incomplete"], "unavailable": counts["unavailable"],
        "event_audit": {"required": [event.value for event in EventType],
                        "covered": [w.event for w in event_witnesses],
                        "missing": [e.value for e in EventType if e.value not in {w.event for w in event_witnesses}]}}
    _atomic_json(directory / "preset_manifest.json", manifest)
    return manifest


def write_event_coverage(preset: Preset, witnesses: Sequence[EventWitness]) -> Path:
    path = preset.output_root / preset.name / "event_coverage.csv"
    _write_csv(path, witnesses, {"preset": preset.name})
    return path


def filter_runs(runs: Sequence[ResolvedRun], *, slots: Iterable[str] = (), seeds: Iterable[int] = ()) -> tuple[ResolvedRun, ...]:
    slot_set, seed_set = set(slots), set(seeds)
    unknown = slot_set - {run.slot_key for run in runs}
    if unknown: raise ValueError(f"unknown slot filters: {sorted(unknown)}")
    return tuple(run for run in runs if (not slot_set or run.slot_key in slot_set)
                 and (not seed_set or run.root_seed in seed_set))


def run_preset(name_or_preset: str | Path | Preset, *, output_root: str | Path | None = None,
               slots: Iterable[str] = (), seeds: Iterable[int] = (), resume: bool = True,
               keep_going: bool = False, confirm_final: bool = False,
               require_clean_git: bool = False, forced_failure_stage: str | None = None) -> RunSummary:
    preset = name_or_preset if isinstance(name_or_preset, Preset) else load_preset(name_or_preset, output_root=output_root)
    if preset.final and not confirm_final: raise PermissionError("final presets require explicit confirmation")
    config, spec = _validate_project(preset); all_runs = expand_preset(preset)
    selected = filter_runs(all_runs, slots=slots, seeds=seeds); provenance = repository_provenance(excluded_paths=(preset.output_root,))
    if require_clean_git and provenance["git"]["dirty"] is not False:
        raise RuntimeError("a clean Git worktree is required")
    witnesses: tuple[EventWitness, ...] = ()
    if preset.event_audit:
        witnesses = audit_event_coverage(MazeMDP(spec, config.rewards))
        write_event_coverage(preset, witnesses)
    complete=skipped=failed=unavailable=0
    for run in selected:
        if not run.available: unavailable += 1; continue
        expected = _run_document(preset, run, config, spec)
        if resume:
            decision = resume_decision(preset.output_root/preset.name/"attempts"/run.slot_id,
                                       expected, spec, provenance)
            if decision.skip: skipped += 1; continue
        try: _execute_attempt(preset, run, config, spec, provenance, forced_failure_stage=forced_failure_stage); complete += 1
        except BaseException:
            failed += 1
            if not keep_going:
                refresh_indexes(preset, all_runs, event_witnesses=witnesses); raise
    refresh_indexes(preset, all_runs, event_witnesses=witnesses)
    return RunSummary(preset.name, complete, skipped, failed, unavailable, len(selected), preset.output_root/preset.name)


def status_preset(name_or_preset: str | Path | Preset, *, output_root: str | Path | None = None) -> Mapping[str, Any]:
    preset = name_or_preset if isinstance(name_or_preset, Preset) else load_preset(name_or_preset, output_root=output_root)
    witnesses: tuple[EventWitness, ...] = ()
    if preset.event_audit:
        config, spec = _validate_project(preset)
        witnesses = audit_event_coverage(MazeMDP(spec, config.rewards))
    return refresh_indexes(preset, expand_preset(preset), event_witnesses=witnesses)


def validate_preset(name_or_preset: str | Path | Preset, *, output_root: str | Path | None = None) -> Mapping[str, Any]:
    preset = name_or_preset if isinstance(name_or_preset, Preset) else load_preset(name_or_preset, output_root=output_root)
    config, spec = _validate_project(preset); provenance = repository_provenance(excluded_paths=(preset.output_root,)); runs=expand_preset(preset)
    valid=invalid=0; errors=[]
    for run in runs:
        if not run.available: continue
        expected=_run_document(preset,run,config,spec)
        for attempt in _attempts(preset.output_root/preset.name/"attempts"/run.slot_id):
            try: validate_attempt(attempt,expected=expected,spec=spec,provenance=provenance); valid+=1
            except ValueError as exc:
                invalid+=1; errors.append({"attempt":str(attempt),"error":str(exc)})
    manifest=refresh_indexes(preset,runs)
    return {"preset":preset.name,"valid_complete_attempts":valid,"invalid_attempts":invalid,
            "errors":errors,"manifest":manifest}


def audit_events(name_or_preset: str | Path | Preset = "smoke", *, output_root: str | Path | None = None) -> tuple[EventWitness, ...]:
    preset=name_or_preset if isinstance(name_or_preset,Preset) else load_preset(name_or_preset,output_root=output_root)
    config,spec=_validate_project(preset); witnesses=audit_event_coverage(MazeMDP(spec,config.rewards))
    write_event_coverage(preset,witnesses); refresh_indexes(preset,expand_preset(preset),event_witnesses=witnesses)
    return witnesses


# CLI-oriented aliases kept deliberately small and stable.
load = load_preset
expand = expand_preset
run = run_preset
status = status_preset
validate = validate_preset

__all__ = [
    "ATTEMPT_SCHEMA_VERSION", "BUILTIN_PRESETS", "CSV_SCHEMA_VERSION", "DEFAULT_OUTPUT_ROOT",
    "EvaluationEpisode", "EvaluationResult", "EvaluationSpec", "EvaluationStep", "EventWitness",
    "FINAL_PRESETS", "MANIFEST_SCHEMA_VERSION", "PRESET_SCHEMA_VERSION", "Preset", "ResolvedRun",
    "ResumeDecision", "RunSummary", "SEED_DERIVATION", "SEED_STREAM_ORDER", "ScientificSlot",
    "SeedStreams", "audit_event_coverage", "audit_events", "canonical_json", "common_rollout_seeds",
    "derive_seed_streams", "evaluate_q_values", "expand", "expand_preset", "filter_runs", "list_presets",
    "load", "load_preset", "refresh_indexes", "repository_provenance", "resume_decision", "run",
    "run_preset", "stable_hash", "status", "status_preset", "validate", "validate_attempt",
    "validate_preset", "write_event_coverage",
]
