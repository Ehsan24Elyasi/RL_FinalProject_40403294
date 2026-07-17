"""Tests for deterministic map generation and canonical serialization."""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path

import pytest

from environments.generator import (
    DEFAULT_BASE_SEED,
    DEFAULT_MAP_PATH,
    DEFAULT_SIZE,
    DEFAULT_STUDENT_ID,
    generate_source_map,
    load_source_map,
    save_source_map,
    source_map_document,
    spec_from_document,
    validate_source_map,
)
from environments.maze import Action, MazeMDP, State


def _states_reachable_without_collecting_key(mdp: MazeMDP) -> set[State]:
    start = mdp.initial_state()
    queue: deque[State] = deque([start])
    visited = {start}
    while queue:
        state = queue.popleft()
        for action in Action:
            for outcome in mdp.transition_outcomes(state, action):
                if outcome.state.has_key or outcome.state in visited:
                    continue
                visited.add(outcome.state)
                queue.append(outcome.state)
    return visited


def test_seed_size_identity_and_determinism() -> None:
    first = generate_source_map()
    second = generate_source_map()

    assert first == second
    assert first.student_id == DEFAULT_STUDENT_ID == "40403294"
    assert first.base_seed == DEFAULT_BASE_SEED == 9
    assert (first.rows, first.cols) == (DEFAULT_SIZE, DEFAULT_SIZE) == (16, 16)
    assert generate_source_map(base_seed=10) != first


def test_phase_one_rejects_other_sizes() -> None:
    with pytest.raises(ValueError, match="requires size 16"):
        generate_source_map(size=15)


def test_committed_source_map_matches_generator_and_round_trips(
    tmp_path: Path,
) -> None:
    generated = generate_source_map()
    committed = load_source_map(DEFAULT_MAP_PATH)
    assert committed == generated

    output = tmp_path / "source.json"
    save_source_map(generated, output)
    loaded = load_source_map(output)
    assert loaded == generated
    assert source_map_document(loaded) == source_map_document(generated)


def test_checksum_detects_tampering() -> None:
    document = source_map_document(generate_source_map())
    document["entities"]["start"] = [1, 2]

    with pytest.raises(ValueError, match="checksum mismatch"):
        spec_from_document(document)


def test_json_has_schema_metadata_checksum_and_single_grid_representation() -> None:
    document = json.loads(DEFAULT_MAP_PATH.read_text(encoding="utf-8"))

    assert document["schema"] == "rl-maze-source-map"
    assert document["schema_version"] == 1
    assert document["metadata"]["student_id"] == "40403294"
    assert document["metadata"]["base_seed"] == 9
    assert document["checksum"]["algorithm"] == "sha256"
    assert len(document["checksum"]["value"]) == 64
    assert "grid" not in document
    assert "ascii" not in document
    assert isinstance(document["walls"], list)


def test_wall_fraction_boundary_and_penalty_nonoverlap() -> None:
    spec = generate_source_map()
    report = validate_source_map(spec)

    expected_boundary = {
        (row, col)
        for row in range(spec.rows)
        for col in range(spec.cols)
        if row in (0, spec.rows - 1) or col in (0, spec.cols - 1)
    }
    assert expected_boundary <= spec.walls
    assert report.interior_wall_fraction >= 0.15
    assert report.interior_wall_fraction == pytest.approx(0.20, abs=0.01)
    assert len(spec.penalties) >= 8
    assert spec.penalties.isdisjoint(spec.walls)
    assert spec.penalties.isdisjoint(
        {
            spec.start,
            spec.goal,
            spec.key,
            spec.door,
            *spec.teleporter_pair,
        }
    )


def test_ordered_mission_and_goal_unreachable_without_key() -> None:
    spec = generate_source_map()
    mdp = MazeMDP(spec)
    report = validate_source_map(spec)

    assert mdp.completion_distance(mdp.initial_state()) == (
        report.completion_distance_with_teleporter
    )
    reachable_without_key = _states_reachable_without_collecting_key(mdp)
    assert State(*spec.key, has_key=False) not in reachable_without_key
    assert all(state.position != spec.goal for state in reachable_without_key)

    key_state = State(*spec.key, has_key=True)
    assert mdp.completion_distance(key_state) is not None


def test_teleporter_is_pre_door_one_hop_non_bypassing_and_consequential() -> None:
    spec = generate_source_map()
    mdp = MazeMDP(spec)
    report = validate_source_map(spec)
    first, second = spec.teleporter_pair

    assert first[1] < spec.door[1]
    assert second[1] < spec.door[1]
    assert report.completion_distance_without_teleporter - (
        report.completion_distance_with_teleporter
    ) >= 4

    # If the door is treated as a wall, no key-holding search can reach goal.
    queue: deque[State] = deque([State(*spec.key, has_key=True)])
    visited = set(queue)
    while queue:
        state = queue.popleft()
        for action in Action:
            for outcome in mdp.transition_outcomes(state, action):
                next_state = outcome.state
                if next_state.position == spec.door or next_state in visited:
                    continue
                visited.add(next_state)
                queue.append(next_state)
    assert all(state.position != spec.goal for state in visited)
