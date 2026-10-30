"""CLI for preparing grouped MMLU clients and running federated LoRA."""

import argparse
import os
from pathlib import Path

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated LoRA pipeline")
    parser.add_argument("action", choices=("prepare", "schedule", "dry-run", "run", "compare", "plot"))
    parser.add_argument("--config", help="YAML configuration; default: MMLU trace FedAsync-LoRA")
    parser.add_argument("--algorithm", choices=("fedit", "ffa-lora", "flora", "fedasync-lora"),
                        help="Select a baseline with its default MMLU session configuration")
    parser.add_argument("--mode", choices=("sync", "async"), help="Override federation.mode")
    parser.add_argument("--gpu", type=int, help="Physical GPU index for this process")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--availability", choices=("trace", "always"), help="Override sessions.availability")
    parser.add_argument("--simulation-only", action="store_true", help="Compare schedules without training or accuracy")
    args = parser.parse_args()
    if args.algorithm and args.config:
        parser.error("Choose --algorithm or --config; the configuration already selects its algorithm")
    if args.simulation_only and args.action != "compare":
        parser.error("--simulation-only is only supported with compare; use dry-run for trace scheduling")
    root = Path(__file__).resolve().parents[1]
    config_path = args.config or f"configs/mmlu-trace-{args.algorithm or 'fedasync-lora'}.yaml"
    config = load_config(root / config_path, args.mode, args.availability)
    if config.get("data", {}).get("dataset") == "mmlu":
        trace = config.get("sessions", {}).get("enabled")
        if args.action not in (("prepare", "run", "schedule", "dry-run", "compare", "plot") if trace else ("prepare", "run")):
            parser.error("Use prepare/run for MMLU; session configurations also support schedule/dry-run/compare/plot")
        if args.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        if args.action == "prepare":
            from .mmlu_data import prepare
            import json
            print(json.dumps(prepare(config, root), indent=2))
        elif args.action == "compare":
            from .trace_comparison import run_comparison
            print(run_comparison(config, root, resume=args.resume, simulation_only=args.simulation_only))
        elif args.action == "plot":
            from .trace_plotting import plot_run, plot_comparison
            comparison = root / config['output_dir'] / 'comparison'
            print(plot_comparison(config, root) if comparison.exists() else plot_run(config, root))
        elif args.action in ("schedule", "dry-run"):
            from .trace_baselines import schedule
            import json
            print(json.dumps(schedule(config, root, simulate=args.action == "dry-run"), indent=2))
        else:
            from .mmlu_runner import run
            print(run(config, root, resume=args.resume))
        return
    if args.action in ("compare", "plot"):
        if args.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        if args.action == "compare":
            from .comparison import run_comparison
            run_comparison(config, root, resume=args.resume, simulation_only=args.simulation_only)
        else:
            from .plotting import plot_comparison
            plot_comparison(config, root)
        return
    if args.action == "prepare":
        from .data import prepare
        import json
        print(json.dumps(prepare(config, root), indent=2))
    elif args.action in ("schedule", "dry-run"):
        if not config.get("sessions", {}).get("enabled"):
            parser.error("schedule and dry-run need a configuration with sessions.enabled: true")
        if args.action == "dry-run":
            from .session_runner import dry_run
            dry_run(config, root)
        else:
            from .sessions import prepare_sessions, write_schedule
            manifest = prepare_sessions(config, root)
            directory = root / config["output_dir"] / "schedule" / manifest["availability"]
            write_schedule(manifest, directory)
            print(f"Prepared {len(manifest['sessions'])} sessions for {len(manifest['clients'])} clients in {directory}")
    else:
        if args.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        from .runner import run
        run(config, root, resume=args.resume)


if __name__ == "__main__":
    main()
