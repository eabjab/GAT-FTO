"""Quantitative evaluation benchmark suite for GAT-FTO exploration agents.

Implements Section 3.5 of SPECS.md:
Evaluates classical baselines (NearestFrontierAgent, UtilityGreedyAgent, MyopicTSPAgent)
and the learned policy (PointerActorCritic / LearnedPolicyAgent) across fixed test seeds
(seed = 10000..10099 by default) and outputs a formatted summary table and JSON report:
1. Total Path Length to 95% and 98% Coverage
2. Exploration Efficiency (Total Uncovered Free Cells / Total Distance Traveled)
3. Backtracking Ratio (fraction of primitive path steps spent on previously visited cells)
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from baselines.classical_agents import (
    BaseExplorationAgent,
    LearnedPolicyAgent,
    MyopicTSPAgent,
    NearestFrontierAgent,
    UtilityGreedyAgent,
)
from config import EnvConfig, ModelConfig
from envs.exploration_env import FrontierExplorationEnv
from models.gat_encoder import GATEncoder
from models.pointer_actor_critic import PointerActorCritic


DEFAULT_START_SEED: int = 10000
DEFAULT_NUM_SEEDS: int = 100
DEFAULT_OUTPUT_JSON: Path = Path("build") / "evaluation_results.json"


def build_default_agents(
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    model_config: ModelConfig | None = None,
    model_seed: int = 42,
) -> dict[str, BaseExplorationAgent]:
    """Instantiate the standard benchmark agent suite.

    Parameters
    ----------
    checkpoint_path:
        Optional path to a saved ``PointerActorCritic`` state dict checkpoint.
    device:
        Torch device string or object (default ``"cpu"``).
    model_config:
        Optional ``ModelConfig`` override for ``PointerActorCritic``.
    model_seed:
        Deterministic RNG seed used when initializing ``PointerActorCritic`` weights.

    Returns
    -------
    dict[str, BaseExplorationAgent]
        Ordered mapping of agent name to ``BaseExplorationAgent`` instance.
    """
    torch.manual_seed(model_seed)
    cfg = model_config if model_config is not None else ModelConfig()
    encoder = GATEncoder(cfg)
    pac_model = PointerActorCritic(
        encoder=encoder,
        hidden_dim=cfg.hidden_dim,
        logit_clip=cfg.logit_clip,
        config=cfg,
    )
    learned_agent = LearnedPolicyAgent(
        model=pac_model,
        checkpoint_path=checkpoint_path,
        device=device,
        config=cfg,
    )

    return {
        "NearestFrontierAgent": NearestFrontierAgent(),
        "UtilityGreedyAgent": UtilityGreedyAgent(lam=4.0),
        "MyopicTSPAgent": MyopicTSPAgent(),
        "PointerActorCritic": learned_agent,
    }


def evaluate_episode(
    env: FrontierExplorationEnv,
    agent: BaseExplorationAgent,
    seed: int,
    deterministic: bool = True,
) -> dict[str, Any]:
    """Run a single evaluation episode for ``agent`` on map ``seed``.

    Parameters
    ----------
    env:
        Configured ``FrontierExplorationEnv`` instance.
    agent:
        Agent implementing ``select_action(graph_data, deterministic=...)``.
    seed:
        Deterministic episode/map seed.
    deterministic:
        Whether the agent selects actions deterministically.

    Returns
    -------
    dict[str, Any]
        Per-episode quantitative metrics dictionary.
    """
    obs, info = env.reset(seed=int(seed))
    terminated = False
    truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        if len(env.frontiers) == 0:
            break
        action = agent.select_action(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

    total_dist = float(env.total_distance)
    uncovered_cells = int(env.uncovered_free_cells)
    total_free = int(env.total_free_cells)
    final_cov = float(env.coverage)

    reached_95 = env.dist_to_95 is not None
    reached_98 = env.dist_to_98 is not None

    # Use exact distance at threshold crossing when reached; fall back to total episode
    # distance traveled when censored by early truncation (e.g. short --max-steps runs).
    path_length_95 = float(env.dist_to_95) if reached_95 else total_dist
    path_length_98 = float(env.dist_to_98) if reached_98 else total_dist

    efficiency = float(env.exploration_efficiency)
    backtracking = float(env.backtracking_ratio)

    return {
        "seed": int(seed),
        "path_length_to_95": path_length_95,
        "path_length_to_98": path_length_98,
        "raw_dist_to_95": float(env.dist_to_95) if reached_95 else None,
        "raw_dist_to_98": float(env.dist_to_98) if reached_98 else None,
        "reached_95": bool(reached_95),
        "reached_98": bool(reached_98),
        "exploration_efficiency": efficiency,
        "backtracking_ratio": backtracking,
        "total_distance": total_dist,
        "uncovered_free_cells": uncovered_cells,
        "total_free_cells": total_free,
        "final_coverage": final_cov,
        "macro_steps": int(env.step_count),
        "primitive_steps": int(env.primitive_steps),
        "backtracking_steps": int(env.backtracking_steps),
        "total_reward": float(total_reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def aggregate_episode_metrics(
    agent_name: str, episode_records: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Aggregate per-episode metrics across test seeds into summary statistics."""
    if not episode_records:
        return {
            "agent": agent_name,
            "num_seeds": 0,
            "path_length_to_95": 0.0,
            "path_length_to_95_std": 0.0,
            "path_length_to_98": 0.0,
            "path_length_to_98_std": 0.0,
            "dist_to_95": 0.0,
            "dist_to_98": 0.0,
            "exploration_efficiency": 0.0,
            "exploration_efficiency_std": 0.0,
            "backtracking_ratio": 0.0,
            "backtracking_ratio_std": 0.0,
            "final_coverage": 0.0,
            "total_distance": 0.0,
            "reached_95_rate": 0.0,
            "reached_98_rate": 0.0,
            "mean_macro_steps": 0.0,
            "episodes": [],
        }

    arr_95 = np.array([ep["path_length_to_95"] for ep in episode_records], dtype=np.float64)
    arr_98 = np.array([ep["path_length_to_98"] for ep in episode_records], dtype=np.float64)
    arr_eff = np.array(
        [ep["exploration_efficiency"] for ep in episode_records], dtype=np.float64
    )
    arr_bt = np.array(
        [ep["backtracking_ratio"] for ep in episode_records], dtype=np.float64
    )
    arr_cov = np.array([ep["final_coverage"] for ep in episode_records], dtype=np.float64)
    arr_dist = np.array([ep["total_distance"] for ep in episode_records], dtype=np.float64)
    arr_r95 = np.array([float(ep["reached_95"]) for ep in episode_records], dtype=np.float64)
    arr_r98 = np.array([float(ep["reached_98"]) for ep in episode_records], dtype=np.float64)
    arr_steps = np.array([ep["macro_steps"] for ep in episode_records], dtype=np.float64)

    mean_95 = float(np.mean(arr_95))
    std_95 = float(np.std(arr_95))
    mean_98 = float(np.mean(arr_98))
    std_98 = float(np.std(arr_98))
    mean_eff = float(np.mean(arr_eff))
    std_eff = float(np.std(arr_eff))
    mean_bt = float(np.mean(arr_bt))
    std_bt = float(np.std(arr_bt))

    return {
        "agent": agent_name,
        "num_seeds": int(len(episode_records)),
        "path_length_to_95": mean_95,
        "path_length_to_95_std": std_95,
        "path_length_to_98": mean_98,
        "path_length_to_98_std": std_98,
        "dist_to_95": mean_95,
        "dist_to_98": mean_98,
        "exploration_efficiency": mean_eff,
        "exploration_efficiency_std": std_eff,
        "backtracking_ratio": mean_bt,
        "backtracking_ratio_std": std_bt,
        "final_coverage": float(np.mean(arr_cov)),
        "total_distance": float(np.mean(arr_dist)),
        "reached_95_rate": float(np.mean(arr_r95)),
        "reached_98_rate": float(np.mean(arr_r98)),
        "mean_macro_steps": float(np.mean(arr_steps)),
        "episodes": list(episode_records),
    }


def evaluate_agents(
    agents: dict[str, BaseExplorationAgent] | None = None,
    seeds: Sequence[int] | None = None,
    start_seed: int = DEFAULT_START_SEED,
    num_seeds: int = DEFAULT_NUM_SEEDS,
    env_config: EnvConfig | None = None,
    max_steps: int | None = None,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    deterministic: bool = True,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate all exploration agents across fixed test seeds and return structured results.

    Parameters
    ----------
    agents:
        Optional mapping of agent names to ``BaseExplorationAgent`` instances. If ``None``,
        instantiates ``NearestFrontierAgent``, ``UtilityGreedyAgent``, ``MyopicTSPAgent``,
        and ``PointerActorCritic`` (``LearnedPolicyAgent``).
    seeds:
        Optional explicit sequence of integer seeds. Overrides ``start_seed`` and ``num_seeds``.
    start_seed:
        Starting test seed (default ``10000``).
    num_seeds:
        Number of consecutive test seeds to evaluate (default ``100``).
    env_config:
        Optional ``EnvConfig`` instance.
    max_steps:
        Optional override for ``env_config.max_episode_steps``.
    checkpoint_path:
        Optional checkpoint path for ``PointerActorCritic``.
    device:
        Torch device string or object.
    deterministic:
        Whether policies act deterministically during evaluation.
    output_json:
        Optional file path to write the JSON benchmark report.

    Returns
    -------
    dict[str, Any]
        Complete benchmark report dictionary.
    """
    cfg = env_config if env_config is not None else EnvConfig()
    if max_steps is not None:
        cfg = replace(cfg, max_episode_steps=int(max_steps))

    if seeds is not None:
        seed_list = [int(s) for s in seeds]
    else:
        seed_list = [int(start_seed + i) for i in range(int(num_seeds))]

    if agents is None:
        agents = build_default_agents(
            checkpoint_path=checkpoint_path,
            device=device,
        )

    env = FrontierExplorationEnv(config=cfg)

    agent_reports: dict[str, dict[str, Any]] = {}
    summary_table_data: dict[str, dict[str, float]] = {}

    for agent_name, agent in agents.items():
        episode_records: list[dict[str, Any]] = []
        for s in seed_list:
            ep_metrics = evaluate_episode(
                env=env,
                agent=agent,
                seed=s,
                deterministic=deterministic,
            )
            episode_records.append(ep_metrics)

        agg = aggregate_episode_metrics(agent_name, episode_records)
        agent_reports[agent_name] = agg
        summary_table_data[agent_name] = {
            "path_length_to_95": agg["path_length_to_95"],
            "path_length_to_95_std": agg["path_length_to_95_std"],
            "path_length_to_98": agg["path_length_to_98"],
            "path_length_to_98_std": agg["path_length_to_98_std"],
            "exploration_efficiency": agg["exploration_efficiency"],
            "exploration_efficiency_std": agg["exploration_efficiency_std"],
            "backtracking_ratio": agg["backtracking_ratio"],
            "backtracking_ratio_std": agg["backtracking_ratio_std"],
            "final_coverage": agg["final_coverage"],
            "total_distance": agg["total_distance"],
            "reached_95_rate": agg["reached_95_rate"],
            "reached_98_rate": agg["reached_98_rate"],
        }

    # Provide LearnedPolicyAgent alias in agents dict if PointerActorCritic is present
    if "PointerActorCritic" in agent_reports and "LearnedPolicyAgent" not in agent_reports:
        agent_reports["LearnedPolicyAgent"] = agent_reports["PointerActorCritic"]

    report: dict[str, Any] = {
        "config": {
            **asdict(cfg),
            "start_seed": int(seed_list[0]) if seed_list else int(start_seed),
            "num_seeds": int(len(seed_list)),
            "seeds": seed_list,
            "deterministic": bool(deterministic),
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        },
        "summary": summary_table_data,
        "agents": agent_reports,
    }

    # Also expose top-level agent keys for convenient lookup
    for agent_name, agg_data in agent_reports.items():
        report[agent_name] = agg_data

    if output_json is not None:
        save_json_report(report, output_json)

    return report


def format_summary_table(report: dict[str, Any]) -> str:
    """Format the quantitative benchmark summary table for stdout display."""
    cfg = report.get("config", {})
    seeds = cfg.get("seeds", [])
    seed_range_str = (
        f"{seeds[0]}..{seeds[-1]} (N={len(seeds)})"
        if seeds
        else f"N={cfg.get('num_seeds', 0)}"
    )
    max_steps = cfg.get("max_episode_steps", "N/A")

    header_title = (
        f"GAT-FTO Quantitative Evaluation Benchmark | Seeds: {seed_range_str} | "
        f"Max Macro-Steps: {max_steps}"
    )
    sep = "=" * 108
    thin_sep = "-" * 108

    col_header = (
        f"{'Agent':<24} | "
        f"{'Path Len 95%':>18} | "
        f"{'Path Len 98%':>18} | "
        f"{'Expl. Efficiency':>20} | "
        f"{'Backtrack Ratio':>18} | "
        f"{'Final Cov':>9}"
    )

    lines = [
        sep,
        header_title,
        sep,
        col_header,
        thin_sep,
    ]

    summary = report.get("summary", {})
    for agent_name, metrics in summary.items():
        p95 = metrics["path_length_to_95"]
        p95_std = metrics["path_length_to_95_std"]
        p98 = metrics["path_length_to_98"]
        p98_std = metrics["path_length_to_98_std"]
        eff = metrics["exploration_efficiency"]
        eff_std = metrics["exploration_efficiency_std"]
        bt = metrics["backtracking_ratio"]
        bt_std = metrics["backtracking_ratio_std"]
        cov = metrics["final_coverage"] * 100.0

        row = (
            f"{agent_name:<24} | "
            f"{p95:8.2f} +/- {p95_std:5.2f} | "
            f"{p98:8.2f} +/- {p98_std:5.2f} | "
            f"{eff:9.3f} +/- {eff_std:5.3f} | "
            f"{bt:7.3f} +/- {bt_std:5.3f} | "
            f"{cov:8.2f}%"
        )
        lines.append(row)

    lines.append(sep)
    return "\n".join(lines)


def save_json_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Serialize the benchmark report dictionary to a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the evaluation benchmark suite."""
    parser = argparse.ArgumentParser(
        description="Quantitative evaluation benchmark suite for GAT-FTO exploration agents."
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=DEFAULT_START_SEED,
        help="Starting test seed (default: 10000).",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=DEFAULT_NUM_SEEDS,
        help="Number of fixed test seeds to evaluate (default: 100 -> 10000..10099).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit list of integer seeds (overrides --start-seed and --num-seeds).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum macro-steps per episode (overrides EnvConfig.max_episode_steps).",
    )
    parser.add_argument(
        "--grid-height",
        type=int,
        default=None,
        help="Optional override for grid height (default: 100).",
    )
    parser.add_argument(
        "--grid-width",
        type=int,
        default=None,
        help="Optional override for grid width (default: 100).",
    )
    parser.add_argument(
        "--sensor-range",
        type=int,
        default=None,
        help="Optional override for sensor raycasting range (default: 20).",
    )
    parser.add_argument(
        "--macro-step-horizon",
        type=int,
        default=None,
        help="Optional override for macro-step horizon (default: 15).",
    )
    parser.add_argument(
        "--coverage-target",
        type=float,
        default=None,
        help="Optional override for terminal coverage fraction (default: 0.98).",
    )
    parser.add_argument(
        "--checkpoint",
        "--checkpoint-path",
        dest="checkpoint_path",
        type=str,
        default=None,
        help="Optional path to trained PointerActorCritic checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for PointerActorCritic evaluation (default: cpu).",
    )
    parser.add_argument(
        "--output-json",
        "--output",
        "-o",
        dest="output_json",
        type=str,
        default=str(DEFAULT_OUTPUT_JSON),
        help="Path to save the output JSON report (default: build/evaluation_results.json).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """CLI entry point for running the GAT-FTO evaluation benchmark."""
    args = parse_args(argv)

    env_cfg = EnvConfig()
    overrides: dict[str, Any] = {}
    if args.max_steps is not None:
        overrides["max_episode_steps"] = int(args.max_steps)
    if args.grid_height is not None:
        overrides["grid_height"] = int(args.grid_height)
    if args.grid_width is not None:
        overrides["grid_width"] = int(args.grid_width)
    if args.sensor_range is not None:
        overrides["sensor_range"] = int(args.sensor_range)
    if args.macro_step_horizon is not None:
        overrides["macro_step_horizon"] = int(args.macro_step_horizon)
    if args.coverage_target is not None:
        overrides["coverage_target"] = float(args.coverage_target)

    if overrides:
        env_cfg = replace(env_cfg, **overrides)

    report = evaluate_agents(
        seeds=args.seeds,
        start_seed=args.start_seed,
        num_seeds=args.num_seeds,
        env_config=env_cfg,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        deterministic=True,
        output_json=args.output_json,
    )

    table_str = format_summary_table(report)
    print(table_str)
    if args.output_json:
        print(f"Saved JSON evaluation report to: {Path(args.output_json)}")

    return report


if __name__ == "__main__":
    main()
