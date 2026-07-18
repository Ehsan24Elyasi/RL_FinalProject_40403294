"""Command-line entry point for the maze, Value Iteration, and Q-Learning."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

from agents.common import load_value_iteration_npz, save_value_iteration_npz
from agents.q_learning import (
    QLearningBundlePaths,
    QLearningConfig,
    QLearningResult,
    build_q_learning_run_identity,
    derive_q_learning_seeds,
    preflight_q_learning_bundle,
    save_q_learning_bundle,
    train_q_learning,
    validate_q_learning_bundle,
)
from agents.value_iteration import (
    ValueIterationConfig,
    ValueIterationConvergenceError,
    compare_policy_invariance,
    value_iteration,
)
from config import (
    DEFAULT_CONFIG_PATH,
    OperationalConfig,
    PlanningRun,
    QLearningRun,
    load_config,
)
from environments.generator import (
    DEFAULT_MAP_PATH,
    generate_source_map,
    load_source_map,
    save_source_map,
    validate_source_map,
)
from environments.maze import Action, MazeEpisode, MazeMDP


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _gamma(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed < 1.0:
        raise argparse.ArgumentTypeError("must be in [0, 1)")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="regenerate source.json")
    generate.add_argument("--output", type=Path, default=DEFAULT_MAP_PATH)

    summary = subparsers.add_parser("summary", help="validate and summarize the map")
    summary.add_argument("--map", type=Path, default=DEFAULT_MAP_PATH)

    smoke = subparsers.add_parser("smoke", help="sample a short seeded episode")
    smoke.add_argument("--map", type=Path, default=DEFAULT_MAP_PATH)
    smoke.add_argument("--seed", type=int, default=9)
    smoke.add_argument("--steps", type=_positive_int, default=8)

    vi = subparsers.add_parser("vi", help="Value Iteration operations")
    vi_subparsers = vi.add_subparsers(dest="vi_command", required=True)

    solve = vi_subparsers.add_parser("solve", help="solve and save one VI model")
    solve.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    solve.add_argument("--reward-mode", choices=("sparse", "shaped"), required=True)
    solve.add_argument("--gamma", type=_gamma, required=True)
    solve.add_argument("--output", type=Path)
    solve.add_argument("--overwrite", action="store_true")

    required = vi_subparsers.add_parser("required", help="run all required VI solves")
    required.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    required.add_argument("--overwrite", action="store_true")

    inspect = vi_subparsers.add_parser("inspect", help="validate and summarize an NPZ model")
    inspect.add_argument("model", type=Path)
    inspect.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)

    invariance = vi_subparsers.add_parser(
        "verify-invariance", help="compare sparse/shaped optimal action sets"
    )
    invariance.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    invariance.add_argument("--sparse", type=Path)
    invariance.add_argument("--shaped", type=Path)

    q = subparsers.add_parser("q", help="Q-Learning operations")
    q_subparsers = q.add_subparsers(dest="q_command", required=True)

    train = q_subparsers.add_parser("train", help="train and save one Q-Learning run")
    train.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    train.add_argument("--reward-mode", choices=("sparse", "shaped"), required=True)
    train.add_argument("--schedule", choices=("linear", "exponential"), required=True)
    train.add_argument("--seed", type=int)
    train.add_argument("--episodes", type=_positive_int)
    train.add_argument("--decay-episodes", type=_positive_int)
    train.add_argument("--audit-episode", type=_positive_int)
    train.add_argument(
        "--output-dir",
        type=Path,
        help="pilot directory containing models/ and raw/ subdirectories",
    )
    train.add_argument("--overwrite", action="store_true")

    q_required = q_subparsers.add_parser(
        "required", help="run the two required shaped 5000-episode trainings"
    )
    q_required.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    q_required.add_argument("--overwrite", action="store_true")

    q_inspect = q_subparsers.add_parser(
        "inspect", help="validate and summarize a Q-Learning NPZ model"
    )
    q_inspect.add_argument("model", type=Path)
    q_inspect.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser


def _print_summary(path: Path) -> None:
    spec = load_source_map(path)
    report = validate_source_map(spec)
    print(f"Map: {path}")
    print(f"Identity: student={spec.student_id}, seed={spec.base_seed}")
    print(f"Dimensions: {spec.rows}x{spec.cols}; non-wall cells={spec.nonwall_cells}")
    print(
        f"Interior walls: {report.interior_wall_count}/"
        f"{report.interior_cell_count} ({report.interior_wall_fraction:.1%})"
    )
    print(
        "Completion distance: "
        f"{report.completion_distance_with_teleporter} with teleporter, "
        f"{report.completion_distance_without_teleporter} without"
    )


def _validate_config_map(config: OperationalConfig):
    spec = load_source_map(config.source_map)
    if spec.student_id != config.student_id or spec.base_seed != config.base_seed:
        raise ValueError("config project identity does not match the source map")
    if (spec.rows, spec.cols) != (config.maze_size, config.maze_size):
        raise ValueError("config maze size does not match the source map")
    if spec.max_steps != 3 * spec.nonwall_cells:
        raise ValueError("environment maximum-step multiplier is not 3")
    return spec


def _solve(config: OperationalConfig, run: PlanningRun, output: Path, overwrite: bool):
    spec = _validate_config_map(config)
    mdp = MazeMDP(
        spec,
        config.rewards,
        gamma=run.gamma,
        use_shaping=run.reward_mode == "shaped",
    )
    vi_config = ValueIterationConfig(
        gamma=run.gamma,
        reward_mode=run.reward_mode,
        theta=config.planning.theta,
        max_sweeps=config.planning.max_sweeps,
        tie_tolerance=config.planning.tie_tolerance,
    )
    result = value_iteration(mdp, vi_config)
    saved = save_value_iteration_npz(
        output,
        values=result.values,
        q_values=result.q_values,
        optimal_action_mask=result.optimal_action_mask,
        valid_mask=result.valid_state_mask,
        reachable_mask=result.reachable_state_mask,
        terminal_mask=result.terminal_state_mask,
        delta_history=result.delta_history,
        metadata=result.metadata(),
        overwrite=overwrite,
    )
    loaded = load_value_iteration_npz(saved, expected_spec=spec)
    print(
        f"Solved {run.reward_mode} gamma={run.gamma:.2f}: "
        f"iterations={result.iterations}, final_delta={result.delta_history[-1]:.3e}, "
        f"residual={result.bellman_residual:.3e}, runtime={result.runtime_seconds:.3f}s"
    )
    print(f"Saved and reloaded: {saved}")
    return result, loaded, saved


def _default_model_path(config: OperationalConfig, run: PlanningRun) -> Path:
    return config.planning.output_dir / run.filename


def _inspect(model: Path, config_path: Path) -> None:
    config = load_config(config_path)
    spec = _validate_config_map(config)
    loaded = load_value_iteration_npz(model, expected_spec=spec)
    metadata = loaded.metadata
    reachable_nonterminal = int(
        (loaded.reachable_state_mask & ~loaded.terminal_state_mask).sum()
    )
    print(f"Model: {model}")
    print(
        f"Mode={metadata['reward_mode']} gamma={float(metadata['gamma']):.2f} "
        f"iterations={int(metadata['iterations'])}"
    )
    print(
        f"Final delta={float(metadata['final_delta']):.3e}; "
        f"Bellman residual={float(metadata['bellman_residual']):.3e}; "
        f"runtime={float(metadata['runtime_seconds']):.3f}s"
    )
    print(
        f"Shape={loaded.values.shape}; reachable nonterminal states={reachable_nonterminal}; "
        f"checksum={metadata['map_checksum']}"
    )


def _q_components(
    config: OperationalConfig,
    run: QLearningRun,
    *,
    decay_episodes: int,
    audit_episode: int,
):
    spec = _validate_config_map(config)
    mdp = MazeMDP(
        spec,
        config.rewards,
        gamma=config.q_learning.gamma,
        use_shaping=run.reward_mode == "shaped",
    )
    q_config = QLearningConfig(
        gamma=config.q_learning.gamma,
        alpha=config.q_learning.alpha,
        episodes=run.episodes,
        shaping_method=config.q_learning.shaping_method,
        shaping_version=config.q_learning.shaping_version,
        epsilon_start=config.q_learning.epsilon_start,
        epsilon_end=config.q_learning.epsilon_end,
        decay_episodes=decay_episodes,
        schedule=run.schedule,
        reward_mode=run.reward_mode,
        audit_episode=audit_episode,
    )
    seeds = derive_q_learning_seeds(run.seed)
    identity = build_q_learning_run_identity(mdp, q_config, seeds)
    return spec, mdp, q_config, identity


def _q_paths(
    config: OperationalConfig,
    run: QLearningRun,
    identity,
    output_dir: Path | None = None,
) -> QLearningBundlePaths:
    if output_dir is None:
        model_dir = config.q_learning.model_dir
        raw_dir = config.q_learning.raw_dir
    else:
        model_dir = output_dir / "models"
        raw_dir = output_dir / "raw"
    stem = f"{run.artifact_stem}_rid_{identity.short_id}"
    return QLearningBundlePaths(
        model=model_dir / f"{stem}.npz",
        episode_metrics=raw_dir / f"{stem}_episodes.csv",
        audit=raw_dir / f"{stem}_audit.csv",
        manifest=raw_dir / f"{stem}_manifest.json",
    )


def _q_summary(result: QLearningResult) -> str:
    metrics = result.episode_metrics
    metadata = result.metadata()
    successes = int(metadata["total_successes"])
    truncations = int(metadata["total_truncated"])
    mean_steps = int(metadata["total_steps"]) / len(metrics)
    mean_learning_return = sum(metric.learning_return for metric in metrics) / len(metrics)
    return (
        f"episodes={len(metrics)}, successes={successes}, truncations={truncations}, "
        f"success_rate={successes / len(metrics):.3f}, mean_steps={mean_steps:.2f}, "
        f"mean_learning_return={mean_learning_return:.6f}, "
        f"runtime={result.runtime_seconds:.3f}s"
    )


def _train_q(
    config: OperationalConfig,
    run: QLearningRun,
    *,
    decay_episodes: int,
    audit_episode: int,
    output_dir: Path | None,
    overwrite: bool,
):
    spec, mdp, q_config, identity = _q_components(
        config,
        run,
        decay_episodes=decay_episodes,
        audit_episode=audit_episode,
    )
    paths = _q_paths(config, run, identity, output_dir)
    preflight_q_learning_bundle(paths, overwrite=overwrite)
    result = train_q_learning(
        mdp, q_config, root_seed=run.seed, identity=identity
    )
    loaded, manifest = save_q_learning_bundle(
        paths,
        result,
        expected_spec=spec,
        overwrite=overwrite,
    )
    print(
        f"Trained Q-Learning {run.reward_mode} {run.schedule} seed={run.seed}: "
        f"{_q_summary(result)}"
    )
    print(f"Run ID: {identity.run_id}")
    print(
        f"Saved and validated bundle model: {paths.model} "
        f"(steps={int(loaded.metadata['total_steps'])})"
    )
    print(f"Episode metrics: {paths.episode_metrics}")
    print(f"Audit trace: {paths.audit} ({len(result.audit_rows)} rows)")
    print(f"Completion manifest: {paths.manifest} (complete={manifest['complete']})")
    return result, loaded, paths


def _inspect_q(model: Path, config_path: Path) -> None:
    config = load_config(config_path)
    spec = _validate_config_map(config)
    manifest_path = model.with_name(model.stem + "_manifest.json")
    if not manifest_path.exists():
        candidate_dirs = [config.q_learning.raw_dir, model.parent.parent / "raw"]
        candidates = sorted(
            {
                candidate.resolve()
                for directory in candidate_dirs
                for candidate in directory.glob(model.stem + "_manifest.json")
            }
        )
        if len(candidates) == 1:
            manifest_path = candidates[0]
    loaded, manifest, paths = validate_q_learning_bundle(
        manifest_path, expected_spec=spec, expected_model=model
    )
    metadata = loaded.metadata
    print(f"Q-Learning model: {paths.model}")
    print(f"Run ID={metadata['run_id']}; manifest complete={manifest['complete']}")
    print(
        f"Mode={metadata['reward_mode']} schedule={metadata['schedule']} "
        f"seed={int(metadata['root_seed'])} episodes={int(metadata['episodes'])}"
    )
    print(
        f"Gamma={float(metadata['gamma']):.2f} alpha={float(metadata['alpha']):.2f} "
        f"epsilon={float(metadata['epsilon_start']):.2f}->"
        f"{float(metadata['epsilon_end']):.2f} through episode "
        f"{int(metadata['decay_episodes'])}"
    )
    print(
        f"Steps={int(metadata['total_steps'])}; successes={int(metadata['total_successes'])}; "
        f"truncations={int(metadata['total_truncated'])}; runtime="
        f"{float(metadata['runtime_seconds']):.3f}s"
    )
    reachable_count = int(loaded.reachable_state_mask.sum())
    visited_count = int(
        (loaded.state_visit_counts > 0)[loaded.reachable_state_mask].sum()
    )
    print(
        f"Shape={loaded.q_values.shape}; state visits={int(metadata['state_visit_total'])}; "
        f"reachable states={reachable_count}; visited reachable={visited_count} "
        f"({visited_count / reachable_count:.1%}); checksum={metadata['map_checksum']}"
    )
    print(f"Manifest: {paths.manifest}")


def _verify_invariance(config_path: Path, sparse_path: Path | None, shaped_path: Path | None) -> bool:
    config = load_config(config_path)
    spec = _validate_config_map(config)
    sparse_path = sparse_path or _default_model_path(config, PlanningRun("sparse", 0.95))
    shaped_path = shaped_path or _default_model_path(config, PlanningRun("shaped", 0.95))
    sparse = load_value_iteration_npz(sparse_path, expected_spec=spec)
    shaped = load_value_iteration_npz(shaped_path, expected_spec=spec)
    if float(sparse.metadata["gamma"]) != 0.95 or float(shaped.metadata["gamma"]) != 0.95:
        raise ValueError("invariance inputs must both use gamma 0.95")
    comparison = (
        sparse.reachable_state_mask
        & shaped.reachable_state_mask
        & ~sparse.terminal_state_mask
    )
    disagreements = comparison & (sparse.optimal_action_mask != shaped.optimal_action_mask).any(axis=-1)
    count = int(disagreements.sum())
    checked = int(comparison.sum())
    print(f"Policy invariance: checked={checked}, disagreements={count}")
    if count:
        first = tuple(int(value) for value in next(zip(*disagreements.nonzero(), strict=True)))
        print(f"First disagreement index: {first}")
    return count == 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            destination = save_source_map(generate_source_map(), args.output)
            print(f"Generated {destination}")
            return 0
        if args.command == "summary":
            _print_summary(args.map)
            return 0
        if args.command == "smoke":
            spec = load_source_map(args.map)
            episode = MazeEpisode(MazeMDP(spec), seed=args.seed)
            policy_rng = random.Random(args.seed)
            state = episode.reset(seed=args.seed)
            print(f"Reset: state={state}, max_steps={episode.max_steps}")
            for _ in range(args.steps):
                if episode.done:
                    break
                action = policy_rng.choice(tuple(Action))
                result = episode.step(action)
                event_names = [event.value for event in result.events]
                print(
                    f"step={result.step_number} intended={action.name} "
                    f"actual={result.actual_action.name} state={result.state} "
                    f"reward={result.total_reward:.2f} events={event_names}"
                )
            print(
                f"Smoke complete: steps={episode.elapsed_steps}, state={episode.state}, "
                f"done={episode.done}"
            )
            return 0

        config = load_config(args.config)
        if args.command == "vi":
            if args.vi_command == "solve":
                run = PlanningRun(args.reward_mode, args.gamma)
                output = args.output or _default_model_path(config, run)
                _solve(config, run, output, args.overwrite)
                return 0
            if args.vi_command == "required":
                results = {}
                for run in config.planning.required_runs:
                    result, _, _ = _solve(
                        config,
                        run,
                        _default_model_path(config, run),
                        args.overwrite,
                    )
                    results[(run.reward_mode, run.gamma)] = result
                invariant, disagreements = compare_policy_invariance(
                    results[("sparse", 0.95)], results[("shaped", 0.95)]
                )
                print(
                    "Policy invariance after required runs: "
                    f"disagreements={int(disagreements.sum())}"
                )
                return 0 if invariant else 2
            if args.vi_command == "inspect":
                _inspect(args.model, args.config)
                return 0
            return 0 if _verify_invariance(
                args.config, args.sparse, args.shaped
            ) else 2

        if args.q_command == "inspect":
            _inspect_q(args.model, args.config)
            return 0
        if args.q_command == "train":
            seed = config.base_seed if args.seed is None else args.seed
            episodes = config.q_learning.episodes if args.episodes is None else args.episodes
            decay_episodes = (
                config.q_learning.decay_episodes
                if args.decay_episodes is None
                else args.decay_episodes
            )
            audit_episode = (
                config.q_learning.audit_episode
                if args.audit_episode is None
                else args.audit_episode
            )
            run = QLearningRun(
                args.reward_mode, args.schedule, seed, episodes
            )
            _train_q(
                config,
                run,
                decay_episodes=decay_episodes,
                audit_episode=audit_episode,
                output_dir=args.output_dir,
                overwrite=args.overwrite,
            )
            return 0

        required_paths = []
        for run in config.q_learning.required_runs:
            _, _, _, identity = _q_components(
                config,
                run,
                decay_episodes=config.q_learning.decay_episodes,
                audit_episode=config.q_learning.audit_episode,
            )
            required_paths.append(_q_paths(config, run, identity))
        for paths in required_paths:
            preflight_q_learning_bundle(paths, overwrite=args.overwrite)
        for run in config.q_learning.required_runs:
            _train_q(
                config,
                run,
                decay_episodes=config.q_learning.decay_episodes,
                audit_episode=config.q_learning.audit_episode,
                output_dir=None,
                overwrite=args.overwrite,
            )
        return 0
    except (OSError, ValueError, FileExistsError, ValueIterationConvergenceError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
