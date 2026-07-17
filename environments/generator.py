"""Deterministic source-map generation and canonical JSON serialization."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

from .maze import Action, Coordinate, MazeMDP, MazeSpec, State

DEFAULT_STUDENT_ID = "40403294"
DEFAULT_BASE_SEED = 9
DEFAULT_SIZE = 16
SCHEMA_NAME = "rl-maze-source-map"
SCHEMA_VERSION = 1
GENERATOR_VERSION = "phase1-deterministic-v1"
DEFAULT_MAP_PATH = Path(__file__).with_name("maps") / "source.json"


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Key measurable invariants of a generated source map."""

    interior_wall_count: int
    interior_cell_count: int
    interior_wall_fraction: float
    completion_distance_with_teleporter: int
    completion_distance_without_teleporter: int
    teleporter_saving: int


def _stable_seed(student_id: str, base_seed: int, size: int) -> int:
    material = f"{student_id}|{base_seed}|{size}|{GENERATOR_VERSION}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _boundary_walls(size: int) -> set[Coordinate]:
    last = size - 1
    walls = {(0, col) for col in range(size)}
    walls.update({(last, col) for col in range(size)})
    walls.update({(row, 0) for row in range(1, last)})
    walls.update({(row, last) for row in range(1, last)})
    return walls


def _protected_corridors() -> set[Coordinate]:
    """Cells preserving a clear ordered mission and teleporter shortcut."""

    protected: set[Coordinate] = set()

    # Direct non-teleporter start-to-key route.
    protected.update((row, 1) for row in range(1, 14))
    protected.add((13, 2))

    # Short teleporter route: start -> A, B -> key.
    protected.update({(1, 2), (2, 2)})
    protected.update((12, col) for col in range(2, 7))

    # Key -> door without touching teleporter B at (12, 6).
    protected.update((13, col) for col in range(2, 8))
    protected.update((row, 7) for row in range(8, 14))
    protected.add((8, 8))

    # Door -> goal on the far side of the separator.
    protected.update((row, 9) for row in range(8, 15))
    protected.update((14, col) for col in range(9, 15))
    return protected


def generate_source_map(
    *,
    student_id: str = DEFAULT_STUDENT_ID,
    base_seed: int = DEFAULT_BASE_SEED,
    size: int = DEFAULT_SIZE,
) -> MazeSpec:
    """Generate the deterministic 16x16 Phase 1 source map.

    The separator topology is fixed so the door is the only crossing between
    mission regions.  The student identifier and base seed deterministically
    select the remaining interior walls and penalty locations.
    """

    if size != DEFAULT_SIZE:
        raise ValueError(f"Phase 1 requires size {DEFAULT_SIZE}, received {size}")
    if not student_id.strip():
        raise ValueError("student_id must not be empty")

    rng = random.Random(_stable_seed(student_id, base_seed, size))
    start = (1, 1)
    key = (13, 2)
    door = (8, 8)
    goal = (14, 14)
    teleporter_pair = ((2, 2), (12, 6))

    walls = _boundary_walls(size)
    # A complete north/south separator, with exactly one non-wall door cell.
    walls.update((row, 8) for row in range(1, size - 1) if (row, 8) != door)

    interior_cells = (size - 2) ** 2
    target_interior_walls = round(interior_cells * 0.20)
    protected = _protected_corridors()
    reserved = {start, key, door, goal, *teleporter_pair}
    candidates = [
        (row, col)
        for row in range(1, size - 1)
        for col in range(1, size - 1)
        if (row, col) not in walls
        and (row, col) not in protected
        and (row, col) not in reserved
    ]
    rng.shuffle(candidates)
    current_interior_walls = sum(
        1
        for row, col in walls
        if 0 < row < size - 1 and 0 < col < size - 1
    )
    walls.update(candidates[: target_interior_walls - current_interior_walls])

    penalty_candidates = [
        (row, col)
        for row in range(1, size - 1)
        for col in range(1, size - 1)
        if (row, col) not in walls and (row, col) not in reserved
    ]
    rng.shuffle(penalty_candidates)
    penalties = frozenset(penalty_candidates[:10])

    spec = MazeSpec(
        rows=size,
        cols=size,
        start=start,
        goal=goal,
        key=key,
        door=door,
        walls=frozenset(walls),
        penalties=penalties,
        teleporter_pair=teleporter_pair,
        student_id=student_id,
        base_seed=base_seed,
        schema_version=SCHEMA_VERSION,
    )
    validate_source_map(spec)
    return spec


def _reachable_positions(
    spec: MazeSpec,
    start: Coordinate,
    *,
    block_door: bool,
    teleporter_enabled: bool,
) -> set[Coordinate]:
    queue: deque[Coordinate] = deque([start])
    visited = {start}
    while queue:
        row, col = queue.popleft()
        for action in Action:
            row_delta, col_delta = action.delta
            candidate = (row + row_delta, col + col_delta)
            if not spec.in_bounds(candidate) or candidate in spec.walls:
                continue
            if block_door and candidate == spec.door:
                continue
            destination = (
                spec.paired_teleporter(candidate)
                if teleporter_enabled
                else None
            )
            next_position = destination or candidate
            if next_position not in visited:
                visited.add(next_position)
                queue.append(next_position)
    return visited


def validate_source_map(spec: MazeSpec) -> ValidationReport:
    """Reject maps that violate Phase 1 topology or mission requirements."""

    if (spec.rows, spec.cols) != (DEFAULT_SIZE, DEFAULT_SIZE):
        raise ValueError("The Phase 1 source map must be exactly 16x16")

    expected_boundary = _boundary_walls(spec.rows)
    missing_boundary = expected_boundary - set(spec.walls)
    if missing_boundary:
        raise ValueError(f"Boundary is not fully walled: {sorted(missing_boundary)}")

    interior_wall_count = sum(
        1
        for row, col in spec.walls
        if 0 < row < spec.rows - 1 and 0 < col < spec.cols - 1
    )
    interior_cell_count = (spec.rows - 2) * (spec.cols - 2)
    interior_wall_fraction = interior_wall_count / interior_cell_count
    if interior_wall_fraction < 0.15:
        raise ValueError("At least 15% of interior cells must be walls")
    if len(spec.penalties) < 8:
        raise ValueError("At least eight penalty cells are required")

    separator_col = spec.door[1]
    expected_separator = {
        (row, separator_col)
        for row in range(1, spec.rows - 1)
        if (row, separator_col) != spec.door
    }
    if not expected_separator.issubset(spec.walls):
        raise ValueError("The separator must be solid except for its door")
    if spec.door in spec.walls:
        raise ValueError("The door must be traversable after key collection")
    if not (spec.start[1] < separator_col and spec.key[1] < separator_col):
        raise ValueError("Start and key must be before the separator door")
    if not all(endpoint[1] < separator_col for endpoint in spec.teleporter_pair):
        raise ValueError("Both teleporters must be before the door")
    if spec.goal[1] <= separator_col:
        raise ValueError("Goal must be after the separator door")

    reachable_before_key = _reachable_positions(
        spec, spec.start, block_door=True, teleporter_enabled=True
    )
    if spec.key not in reachable_before_key:
        raise ValueError("Key is not reachable from the start")
    if spec.goal in reachable_before_key:
        raise ValueError("Goal is reachable without collecting the key")

    # Blocking the door must disconnect the goal even for a key-holding agent;
    # this proves the teleporter cannot bypass the separator.
    reachable_with_door_blocked = _reachable_positions(
        spec, spec.key, block_door=True, teleporter_enabled=True
    )
    if spec.goal in reachable_with_door_blocked:
        raise ValueError("Goal can be reached without traversing the door")

    mdp = MazeMDP(spec)
    start_state = State(*spec.start, has_key=False)
    with_teleporter = mdp.completion_distance(
        start_state, teleporter_enabled=True
    )
    without_teleporter = mdp.completion_distance(
        start_state, teleporter_enabled=False
    )
    if with_teleporter is None:
        raise ValueError("Ordered start -> key -> door -> goal mission is unreachable")
    if without_teleporter is None:
        raise ValueError("Mission must remain reachable without teleportation")
    teleporter_saving = without_teleporter - with_teleporter
    if teleporter_saving < 4:
        raise ValueError(
            "Teleporter must shorten valid completion distance by at least four steps"
        )

    return ValidationReport(
        interior_wall_count=interior_wall_count,
        interior_cell_count=interior_cell_count,
        interior_wall_fraction=interior_wall_fraction,
        completion_distance_with_teleporter=with_teleporter,
        completion_distance_without_teleporter=without_teleporter,
        teleporter_saving=teleporter_saving,
    )


def _coordinates(values: Iterable[Iterable[int]]) -> frozenset[Coordinate]:
    return frozenset((int(row), int(col)) for row, col in values)


def _payload_from_spec(spec: MazeSpec) -> dict[str, Any]:
    return {
        "schema": SCHEMA_NAME,
        "schema_version": spec.schema_version,
        "metadata": {
            "student_id": spec.student_id,
            "base_seed": spec.base_seed,
            "generator_version": GENERATOR_VERSION,
        },
        "dimensions": {"rows": spec.rows, "cols": spec.cols},
        "entities": {
            "start": list(spec.start),
            "goal": list(spec.goal),
            "key": list(spec.key),
            "door": list(spec.door),
            "teleporter_pair": [list(point) for point in spec.teleporter_pair],
        },
        # Coordinates are the only grid representation; there is no duplicate
        # ASCII or matrix form that could become inconsistent.
        "walls": [list(point) for point in sorted(spec.walls)],
        "penalties": [list(point) for point in sorted(spec.penalties)],
    }


def canonical_payload_bytes(payload: Mapping[str, Any]) -> bytes:
    """Canonical UTF-8 representation used by the map checksum."""

    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def source_map_checksum(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_payload_bytes(payload)).hexdigest()


def source_map_document(spec: MazeSpec) -> dict[str, Any]:
    payload = _payload_from_spec(spec)
    return {
        **payload,
        "checksum": {
            "algorithm": "sha256",
            "value": source_map_checksum(payload),
        },
    }


def spec_from_document(document: Mapping[str, Any]) -> MazeSpec:
    """Verify schema/checksum and reconstruct an immutable :class:`MazeSpec`."""

    if document.get("schema") != SCHEMA_NAME:
        raise ValueError(f"Unsupported source-map schema: {document.get('schema')!r}")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema version: {document.get('schema_version')!r}"
        )
    checksum = document.get("checksum")
    if not isinstance(checksum, Mapping) or checksum.get("algorithm") != "sha256":
        raise ValueError("Source map must contain a sha256 checksum")

    payload = {key: value for key, value in document.items() if key != "checksum"}
    expected = source_map_checksum(payload)
    if checksum.get("value") != expected:
        raise ValueError("Source-map checksum mismatch")

    metadata = payload["metadata"]
    if metadata.get("generator_version") != GENERATOR_VERSION:
        raise ValueError("Unsupported generator version")
    dimensions = payload["dimensions"]
    entities = payload["entities"]
    teleporter_values = entities["teleporter_pair"]
    if len(teleporter_values) != 2:
        raise ValueError("Exactly two teleporter endpoints are required")

    spec = MazeSpec(
        rows=int(dimensions["rows"]),
        cols=int(dimensions["cols"]),
        start=tuple(entities["start"]),
        goal=tuple(entities["goal"]),
        key=tuple(entities["key"]),
        door=tuple(entities["door"]),
        walls=_coordinates(payload["walls"]),
        penalties=_coordinates(payload["penalties"]),
        teleporter_pair=(
            tuple(teleporter_values[0]),
            tuple(teleporter_values[1]),
        ),
        student_id=str(metadata["student_id"]),
        base_seed=int(metadata["base_seed"]),
        schema_version=int(payload["schema_version"]),
    )
    validate_source_map(spec)
    return spec


def save_source_map(spec: MazeSpec, path: Path | str = DEFAULT_MAP_PATH) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = source_map_document(spec)
    destination.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def load_source_map(path: Path | str = DEFAULT_MAP_PATH) -> MazeSpec:
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read source map {source}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise ValueError("Source-map root must be a JSON object")
    return spec_from_document(document)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-id", default=DEFAULT_STUDENT_ID)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--output", type=Path, default=DEFAULT_MAP_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    spec = generate_source_map(
        student_id=args.student_id, base_seed=args.base_seed, size=args.size
    )
    output = save_source_map(spec, args.output)
    report = validate_source_map(spec)
    print(f"Wrote canonical source map: {output}")
    print(
        "Completion distance: "
        f"{report.completion_distance_with_teleporter} with teleporter, "
        f"{report.completion_distance_without_teleporter} without "
        f"(saving {report.teleporter_saving})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
