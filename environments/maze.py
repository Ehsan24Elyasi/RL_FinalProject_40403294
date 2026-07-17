"""Core Markov decision process and episode wrapper for the Phase 1 maze.

The :class:`MazeMDP` owns the deterministic transition semantics for an actual
movement direction.  Both planning code and :class:`MazeEpisode` consume the
same ``transition_outcomes`` method, so sampled interaction cannot drift from
the model used by later dynamic-programming phases.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
import random
from typing import Callable, TypeAlias

Coordinate: TypeAlias = tuple[int, int]
PotentialFunction: TypeAlias = Callable[["State"], float]


class Action(Enum):
    """Cardinal movement actions in stable clockwise order."""

    UP = (-1, 0)
    RIGHT = (0, 1)
    DOWN = (1, 0)
    LEFT = (0, -1)

    @property
    def delta(self) -> Coordinate:
        return self.value

    @property
    def perpendicular(self) -> tuple["Action", "Action"]:
        """Return the two possible slip directions in stable order."""

        if self in (Action.UP, Action.DOWN):
            return (Action.LEFT, Action.RIGHT)
        return (Action.UP, Action.DOWN)

    @classmethod
    def parse(cls, value: "Action | str") -> "Action":
        if isinstance(value, cls):
            return value
        try:
            return cls[value.strip().upper()]
        except (KeyError, AttributeError) as exc:
            valid = ", ".join(action.name for action in cls)
            raise ValueError(f"Unknown action {value!r}; expected one of: {valid}") from exc


class EventType(Enum):
    """Events emitted in the order that transition semantics are applied."""

    MOVE = "move"
    WALL_COLLISION = "wall_collision"
    PENALTY_ENTERED = "penalty_entered"
    KEY_COLLECTED = "key_collected"
    CLOSED_DOOR_ATTEMPT = "closed_door_attempt"
    DOOR_PASSED = "door_passed"
    TELEPORTED = "teleported"
    GOAL_REACHED = "goal_reached"
    EPISODE_TRUNCATED = "episode_truncated"


@dataclass(frozen=True, slots=True, order=True)
class State:
    """Minimal Markov state: location and whether the key was collected."""

    row: int
    col: int
    has_key: bool = False

    @property
    def position(self) -> Coordinate:
        return (self.row, self.col)


@dataclass(frozen=True, slots=True)
class MazeSpec:
    """Immutable structural definition of one maze."""

    rows: int
    cols: int
    start: Coordinate
    goal: Coordinate
    key: Coordinate
    door: Coordinate
    walls: frozenset[Coordinate]
    penalties: frozenset[Coordinate]
    teleporter_pair: tuple[Coordinate, Coordinate]
    student_id: str = "40403294"
    base_seed: int = 9
    schema_version: int = 1

    def __post_init__(self) -> None:
        # Frozen dataclasses prevent attribute reassignment, but callers could
        # still pass mutable sets/lists. Normalize containers so the instance is
        # deeply immutable at the public boundary.
        object.__setattr__(self, "start", tuple(self.start))
        object.__setattr__(self, "goal", tuple(self.goal))
        object.__setattr__(self, "key", tuple(self.key))
        object.__setattr__(self, "door", tuple(self.door))
        object.__setattr__(self, "walls", frozenset(map(tuple, self.walls)))
        object.__setattr__(self, "penalties", frozenset(map(tuple, self.penalties)))
        object.__setattr__(
            self,
            "teleporter_pair",
            tuple(map(tuple, self.teleporter_pair)),
        )

        if self.rows < 3 or self.cols < 3:
            raise ValueError("Maze dimensions must both be at least 3")
        if len(self.teleporter_pair) != 2:
            raise ValueError("Exactly two teleporter endpoints are required")
        if self.teleporter_pair[0] == self.teleporter_pair[1]:
            raise ValueError("Teleporter endpoints must be distinct")

        named = {
            "start": self.start,
            "goal": self.goal,
            "key": self.key,
            "door": self.door,
            "teleporter A": self.teleporter_pair[0],
            "teleporter B": self.teleporter_pair[1],
        }
        if len(set(named.values())) != len(named):
            raise ValueError("Start, goal, key, door, and teleporters must not overlap")

        for name, position in named.items():
            if not self.in_bounds(position):
                raise ValueError(f"{name} coordinate is out of bounds: {position}")
            if position in self.walls:
                raise ValueError(f"{name} coordinate cannot be a wall: {position}")

        for position in self.walls | self.penalties:
            if not self.in_bounds(position):
                raise ValueError(f"Coordinate is out of bounds: {position}")

        reserved = set(named.values())
        overlaps = reserved & set(self.penalties)
        if overlaps:
            raise ValueError(f"Penalties overlap reserved entities: {sorted(overlaps)}")
        wall_penalty_overlap = set(self.walls) & set(self.penalties)
        if wall_penalty_overlap:
            raise ValueError(
                f"Penalties overlap walls: {sorted(wall_penalty_overlap)}"
            )

    @property
    def nonwall_cells(self) -> int:
        return self.rows * self.cols - len(self.walls)

    @property
    def max_steps(self) -> int:
        return 3 * self.nonwall_cells

    def in_bounds(self, position: Coordinate) -> bool:
        row, col = position
        return 0 <= row < self.rows and 0 <= col < self.cols

    def paired_teleporter(self, position: Coordinate) -> Coordinate | None:
        first, second = self.teleporter_pair
        if position == first:
            return second
        if position == second:
            return first
        return None


@dataclass(frozen=True, slots=True)
class RewardSpec:
    """Sparse reward coefficients and optional shaping scale."""

    step: float = -0.1
    collision: float = -0.9
    penalty: float = -1.9
    key: float = 5.0
    goal: float = 20.0
    shaping_scale: float = 1.0


@dataclass(frozen=True, slots=True)
class TransitionOutcome:
    """One stochastic branch returned by :meth:`MazeMDP.transition_outcomes`."""

    probability: float
    intended_action: Action
    actual_action: Action
    state: State
    events: tuple[EventType, ...]
    base_reward: float
    shaping_reward: float
    total_reward: float
    terminated: bool


@dataclass(frozen=True, slots=True)
class StepResult:
    """Immutable result of one sampled episode step."""

    state: State
    intended_action: Action
    actual_action: Action
    probability: float
    events: tuple[EventType, ...]
    base_reward: float
    shaping_reward: float
    total_reward: float
    terminated: bool
    truncated: bool
    step_number: int

    @property
    def reward(self) -> float:
        """Gym-style alias for the total reward."""

        return self.total_reward


class MazeMDP:
    """Pure stochastic transition model for an immutable maze specification."""

    INTENDED_PROBABILITY = 0.8
    SLIP_PROBABILITY = 0.1

    def __init__(
        self,
        spec: MazeSpec,
        rewards: RewardSpec | None = None,
        *,
        gamma: float = 0.95,
        use_shaping: bool = False,
        potential_fn: PotentialFunction | None = None,
    ) -> None:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1 inclusive")
        self.spec = spec
        self.rewards = rewards or RewardSpec()
        self.gamma = float(gamma)
        self.use_shaping = use_shaping
        self._potential_fn = potential_fn
        self._default_potentials: dict[State, float] | None = None
        if use_shaping and potential_fn is None:
            self._default_potentials = self._build_default_potentials()

    def initial_state(self) -> State:
        return State(*self.spec.start, has_key=False)

    def transition_outcomes(
        self, state: State, action: Action | str
    ) -> tuple[TransitionOutcome, ...]:
        """Return all three branches without sampling or mutating any state."""

        self._validate_state(state)
        intended = Action.parse(action)
        if state.position == self.spec.goal:
            raise ValueError("The goal is terminal and has no outgoing transitions")

        actual_actions = (intended, *intended.perpendicular)
        probabilities = (
            self.INTENDED_PROBABILITY,
            self.SLIP_PROBABILITY,
            self.SLIP_PROBABILITY,
        )
        return tuple(
            self._outcome_for_actual(state, intended, actual, probability)
            for actual, probability in zip(actual_actions, probabilities, strict=True)
        )

    def completion_distance(
        self,
        state: State,
        *,
        teleporter_enabled: bool = True,
    ) -> int | None:
        """Shortest valid steps from ``state`` to the goal, or ``None``.

        The search augments position with ``has_key`` and applies the same
        blocking/key/teleporter semantics as the transition model.  Stochastic
        probabilities are intentionally ignored: each cardinal actual action is
        treated as an available edge.
        """

        self._validate_state(state)
        queue: deque[tuple[State, int]] = deque([(state, 0)])
        visited = {state}

        while queue:
            current, distance = queue.popleft()
            if current.position == self.spec.goal:
                return distance
            for actual in Action:
                next_state, _, _ = self._apply_actual_direction(
                    current, actual, teleporter_enabled=teleporter_enabled
                )
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_state, distance + 1))
        return None

    def potential(self, state: State) -> float:
        """Evaluate the configured or normalized completion-distance potential."""

        self._validate_state(state)
        if self._potential_fn is not None:
            return float(self._potential_fn(state))
        if self._default_potentials is None:
            self._default_potentials = self._build_default_potentials()
        return self._default_potentials[state]

    def _build_default_potentials(self) -> dict[State, float]:
        """Precompute potentials once so planning does not run BFS per backup."""

        distances: dict[State, int | None] = {}
        for row in range(self.spec.rows):
            for col in range(self.spec.cols):
                if (row, col) in self.spec.walls:
                    continue
                for has_key in (False, True):
                    state = State(row, col, has_key)
                    distances[state] = self.completion_distance(state)

        finite_distances = [distance for distance in distances.values() if distance is not None]
        if not finite_distances:
            raise ValueError("Maze has no state with a finite completion distance")
        maximum_distance = max(finite_distances) or 1
        return {
            state: (-1.0 if distance is None else -distance / maximum_distance)
            for state, distance in distances.items()
        }

    def _outcome_for_actual(
        self,
        source: State,
        intended: Action,
        actual: Action,
        probability: float,
    ) -> TransitionOutcome:
        next_state, events, base_reward = self._apply_actual_direction(source, actual)
        terminated = next_state.position == self.spec.goal
        shaping_reward = 0.0
        if self.use_shaping:
            shaping_reward = self.rewards.shaping_scale * (
                self.gamma * self.potential(next_state) - self.potential(source)
            )
        total_reward = base_reward + shaping_reward
        return TransitionOutcome(
            probability=probability,
            intended_action=intended,
            actual_action=actual,
            state=next_state,
            events=events,
            base_reward=base_reward,
            shaping_reward=shaping_reward,
            total_reward=total_reward,
            terminated=terminated,
        )

    def _apply_actual_direction(
        self,
        source: State,
        actual: Action,
        *,
        teleporter_enabled: bool = True,
    ) -> tuple[State, tuple[EventType, ...], float]:
        """Apply one actual direction in the documented semantic order."""

        row_delta, col_delta = actual.delta
        candidate = (source.row + row_delta, source.col + col_delta)
        events: list[EventType] = []
        base_reward = self.rewards.step

        if not self.spec.in_bounds(candidate) or candidate in self.spec.walls:
            events.append(EventType.WALL_COLLISION)
            base_reward += self.rewards.collision
            return source, tuple(events), base_reward

        if candidate == self.spec.door and not source.has_key:
            events.append(EventType.CLOSED_DOOR_ATTEMPT)
            base_reward += self.rewards.collision
            return source, tuple(events), base_reward

        # Movement occurs before tile effects.
        position = candidate
        has_key = source.has_key
        events.append(EventType.MOVE)

        # The key reward is awarded once because has_key is persistent.
        if position == self.spec.key and not has_key:
            has_key = True
            events.append(EventType.KEY_COLLECTED)
            base_reward += self.rewards.key

        # Penalty applies to the tile entered before any teleportation.
        if position in self.spec.penalties:
            events.append(EventType.PENALTY_ENTERED)
            base_reward += self.rewards.penalty

        # A single pair lookup prevents chaining back to the first endpoint.
        if teleporter_enabled:
            destination = self.spec.paired_teleporter(position)
            if destination is not None:
                position = destination
                events.append(EventType.TELEPORTED)

        # Door and goal checks use the final position after teleportation.
        if position == self.spec.door:
            events.append(EventType.DOOR_PASSED)
        if position == self.spec.goal:
            events.append(EventType.GOAL_REACHED)
            base_reward += self.rewards.goal

        return State(*position, has_key=has_key), tuple(events), base_reward

    def _validate_state(self, state: State) -> None:
        if not self.spec.in_bounds(state.position):
            raise ValueError(f"State is out of bounds: {state}")
        if state.position in self.spec.walls:
            raise ValueError(f"State cannot occupy a wall: {state}")
        if not isinstance(state.has_key, bool):
            raise TypeError("state.has_key must be a bool")


class MazeEpisode:
    """Seeded sampling wrapper around the pure :class:`MazeMDP` model."""

    def __init__(
        self,
        mdp: MazeMDP,
        *,
        seed: int | None = None,
        max_steps: int | None = None,
    ) -> None:
        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.mdp = mdp
        self.max_steps = max_steps or mdp.spec.max_steps
        self._rng = random.Random(seed)
        self._state = mdp.initial_state()
        self._elapsed_steps = 0
        self._terminated = False
        self._truncated = False

    @property
    def state(self) -> State:
        return self._state

    @property
    def elapsed_steps(self) -> int:
        return self._elapsed_steps

    @property
    def done(self) -> bool:
        return self._terminated or self._truncated

    def reset(self, *, seed: int | None = None) -> State:
        if seed is not None:
            self._rng.seed(seed)
        self._state = self.mdp.initial_state()
        self._elapsed_steps = 0
        self._terminated = False
        self._truncated = False
        return self._state

    def step(self, action: Action | str) -> StepResult:
        if self.done:
            raise RuntimeError("Episode is done; call reset() before stepping again")

        intended = Action.parse(action)
        outcomes = self.mdp.transition_outcomes(self._state, intended)
        draw = self._rng.random()
        cumulative = 0.0
        selected = outcomes[-1]
        for outcome in outcomes:
            cumulative += outcome.probability
            if draw < cumulative or math.isclose(cumulative, 1.0):
                selected = outcome
                break

        self._elapsed_steps += 1
        self._state = selected.state
        self._terminated = selected.terminated
        # Reaching the goal on the limit is termination, never truncation.
        self._truncated = (
            not self._terminated and self._elapsed_steps >= self.max_steps
        )

        events = selected.events
        if self._truncated:
            events = (*events, EventType.EPISODE_TRUNCATED)

        return StepResult(
            state=selected.state,
            intended_action=selected.intended_action,
            actual_action=selected.actual_action,
            probability=selected.probability,
            events=events,
            base_reward=selected.base_reward,
            shaping_reward=selected.shaping_reward,
            total_reward=selected.total_reward,
            terminated=self._terminated,
            truncated=self._truncated,
            step_number=self._elapsed_steps,
        )
