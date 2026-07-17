"""Tests for pure MDP semantics and seeded episode sampling."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from environments.maze import (
    Action,
    EventType,
    MazeEpisode,
    MazeMDP,
    MazeSpec,
    RewardSpec,
    State,
)


def _boundary(size: int) -> frozenset[tuple[int, int]]:
    return frozenset(
        (row, col)
        for row in range(size)
        for col in range(size)
        if row in (0, size - 1) or col in (0, size - 1)
    )


def _semantic_spec() -> MazeSpec:
    return MazeSpec(
        rows=7,
        cols=7,
        start=(1, 1),
        goal=(5, 5),
        key=(1, 3),
        door=(3, 3),
        walls=_boundary(7),
        penalties=frozenset({(1, 2)}),
        teleporter_pair=((2, 1), (4, 4)),
    )


def _outcome(mdp: MazeMDP, state: State, action: Action) -> object:
    return next(
        outcome
        for outcome in mdp.transition_outcomes(state, action)
        if outcome.actual_action is action
    )


def test_transition_probabilities_and_perpendicular_actions() -> None:
    mdp = MazeMDP(_semantic_spec())
    outcomes = mdp.transition_outcomes(mdp.initial_state(), Action.UP)

    assert [outcome.probability for outcome in outcomes] == [0.8, 0.1, 0.1]
    assert [outcome.actual_action for outcome in outcomes] == [
        Action.UP,
        Action.LEFT,
        Action.RIGHT,
    ]
    assert sum(outcome.probability for outcome in outcomes) == pytest.approx(1.0)
    assert all(outcome.intended_action is Action.UP for outcome in outcomes)


def test_collision_and_closed_door_leave_state_unchanged() -> None:
    mdp = MazeMDP(_semantic_spec())
    start = mdp.initial_state()
    collision = _outcome(mdp, start, Action.LEFT)

    assert collision.state == start
    assert collision.events == (EventType.WALL_COLLISION,)
    assert collision.base_reward == pytest.approx(-1.0)

    before_door = State(3, 2, has_key=False)
    blocked = _outcome(mdp, before_door, Action.RIGHT)
    assert blocked.state == before_door
    assert blocked.events == (EventType.CLOSED_DOOR_ATTEMPT,)
    assert blocked.base_reward == pytest.approx(-1.0)


def test_key_penalty_door_and_goal_rewards_and_event_order() -> None:
    mdp = MazeMDP(_semantic_spec())

    penalty = _outcome(mdp, State(1, 1), Action.RIGHT)
    assert penalty.state == State(1, 2, has_key=False)
    assert penalty.events == (EventType.MOVE, EventType.PENALTY_ENTERED)
    assert penalty.base_reward == pytest.approx(-2.0)

    key = _outcome(mdp, State(1, 2), Action.RIGHT)
    assert key.state == State(1, 3, has_key=True)
    assert key.events == (EventType.MOVE, EventType.KEY_COLLECTED)
    assert key.base_reward == pytest.approx(4.9)

    revisited_key = _outcome(mdp, State(1, 2, has_key=True), Action.RIGHT)
    assert revisited_key.events == (EventType.MOVE,)
    assert revisited_key.base_reward == pytest.approx(-0.1)

    door = _outcome(mdp, State(3, 2, has_key=True), Action.RIGHT)
    assert door.state == State(3, 3, has_key=True)
    assert door.events == (EventType.MOVE, EventType.DOOR_PASSED)
    assert door.base_reward == pytest.approx(-0.1)

    goal = _outcome(mdp, State(5, 4, has_key=True), Action.RIGHT)
    assert goal.state == State(5, 5, has_key=True)
    assert goal.events == (EventType.MOVE, EventType.GOAL_REACHED)
    assert goal.base_reward == pytest.approx(19.9)
    assert goal.terminated is True


def test_teleporter_is_bidirectional_one_hop_and_has_no_bonus() -> None:
    mdp = MazeMDP(_semantic_spec())

    forward = _outcome(mdp, State(1, 1), Action.DOWN)
    assert forward.state == State(4, 4, has_key=False)
    assert forward.events == (EventType.MOVE, EventType.TELEPORTED)
    assert forward.base_reward == pytest.approx(-0.1)

    reverse = _outcome(mdp, State(4, 3), Action.RIGHT)
    assert reverse.state == State(2, 1, has_key=False)
    assert reverse.events == (EventType.MOVE, EventType.TELEPORTED)
    # It remains on the paired destination instead of recursively bouncing back.
    assert reverse.state.position != (4, 4)


def test_potential_shaping_components_follow_gamma_formula() -> None:
    rewards = RewardSpec(shaping_scale=2.0)
    mdp = MazeMDP(
        _semantic_spec(),
        rewards,
        gamma=0.9,
        use_shaping=True,
        potential_fn=lambda state: float(state.row + state.col),
    )
    source = State(1, 1)
    outcome = _outcome(mdp, source, Action.RIGHT)

    expected_shaping = 2.0 * (0.9 * 3.0 - 2.0)
    assert outcome.base_reward == pytest.approx(-2.0)
    assert outcome.shaping_reward == pytest.approx(expected_shaping)
    assert outcome.total_reward == pytest.approx(-2.0 + expected_shaping)


def test_default_potential_is_normalized_and_terminal_is_zero() -> None:
    mdp = MazeMDP(_semantic_spec(), use_shaping=True)

    assert mdp.potential(State(*mdp.spec.goal, has_key=True)) == pytest.approx(0.0)
    assert -1.0 <= mdp.potential(mdp.initial_state()) < 0.0
    assert mdp.potential(State(5, 4, has_key=True)) > mdp.potential(
        mdp.initial_state()
    )


def test_transition_outcomes_are_pure_and_domain_objects_are_immutable() -> None:
    mdp = MazeMDP(_semantic_spec())
    source = State(1, 1)

    first = mdp.transition_outcomes(source, Action.RIGHT)
    second = mdp.transition_outcomes(source, Action.RIGHT)
    assert first == second
    assert source == State(1, 1, has_key=False)
    assert mdp.initial_state() == source

    with pytest.raises(FrozenInstanceError):
        source.row = 2  # type: ignore[misc]
    with pytest.raises(AttributeError):
        mdp.spec.walls.add((2, 2))  # type: ignore[attr-defined]


def test_maze_spec_normalizes_mutable_inputs_to_immutable_containers() -> None:
    walls = set(_boundary(7))
    penalties = {(1, 2)}
    teleporters = [(2, 1), (4, 4)]
    spec = MazeSpec(
        rows=7,
        cols=7,
        start=[1, 1],  # type: ignore[arg-type]
        goal=[5, 5],  # type: ignore[arg-type]
        key=[1, 3],  # type: ignore[arg-type]
        door=[3, 3],  # type: ignore[arg-type]
        walls=walls,  # type: ignore[arg-type]
        penalties=penalties,  # type: ignore[arg-type]
        teleporter_pair=teleporters,  # type: ignore[arg-type]
    )

    walls.add((2, 2))
    penalties.add((2, 3))
    teleporters[0] = (3, 1)
    assert (2, 2) not in spec.walls
    assert (2, 3) not in spec.penalties
    assert spec.teleporter_pair == ((2, 1), (4, 4))


def _limit_spec() -> MazeSpec:
    return MazeSpec(
        rows=5,
        cols=5,
        start=(1, 1),
        goal=(1, 2),
        key=(2, 1),
        door=(2, 2),
        walls=_boundary(5),
        penalties=frozenset(),
        teleporter_pair=((3, 1), (3, 2)),
    )


def test_goal_termination_precedes_truncation_on_final_allowed_step() -> None:
    episode = MazeEpisode(MazeMDP(_limit_spec()), seed=1, max_steps=1)
    result = episode.step(Action.RIGHT)

    assert result.actual_action is Action.RIGHT
    assert result.terminated is True
    assert result.truncated is False
    assert episode.done is True
    with pytest.raises(RuntimeError, match="call reset"):
        episode.step(Action.RIGHT)


def test_time_limit_truncates_nonterminal_transition() -> None:
    episode = MazeEpisode(MazeMDP(_limit_spec()), seed=1, max_steps=1)
    result = episode.step(Action.LEFT)

    assert result.actual_action is Action.LEFT
    assert result.terminated is False
    assert result.truncated is True
    assert result.state == State(1, 1)
    assert result.events == (
        EventType.WALL_COLLISION,
        EventType.EPISODE_TRUNCATED,
    )


def test_default_limit_is_three_times_nonwall_cells() -> None:
    spec = _semantic_spec()
    episode = MazeEpisode(MazeMDP(spec))
    assert episode.max_steps == 3 * spec.nonwall_cells


def test_seeded_sampling_is_reproducible_across_episodes_and_resets() -> None:
    mdp = MazeMDP(_semantic_spec())
    actions = [
        Action.UP,
        Action.RIGHT,
        Action.DOWN,
        Action.LEFT,
        Action.UP,
        Action.RIGHT,
    ]
    first = MazeEpisode(mdp, seed=1234)
    second = MazeEpisode(mdp, seed=1234)

    first_results = [first.step(action) for action in actions]
    second_results = [second.step(action) for action in actions]
    assert first_results == second_results

    first.reset(seed=77)
    replay_one = [first.step(Action.UP) for _ in range(4)]
    first.reset(seed=77)
    replay_two = [first.step(Action.UP) for _ in range(4)]
    assert replay_one == replay_two
