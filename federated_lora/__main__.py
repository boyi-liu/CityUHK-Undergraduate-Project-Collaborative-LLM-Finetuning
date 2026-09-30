"""CLI for preparing Alpaca clients and running federated LoRA experiments."""

import argparse
import os
from pathlib import Path

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated LoRA pipeline")
    parser.add_argument("action", choices=("prepare", "schedule", "dry-run", "run"))
    parser.add_argument("--config", default="configs/federated-lora-smoke.yaml")
    parser.add_argument("--mode", choices=("sync", "async"), help="Override federation.mode")
    parser.add_argument("--gpu", type=int, help="Physical GPU index for this process")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / args.config, args.mode)
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
            directory = root / config["output_dir"] / "schedule"
            write_schedule(manifest, directory)
            print(f"Prepared {len(manifest['sessions'])} sessions for {len(manifest['clients'])} clients in {directory}")
    else:
        if args.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        from .runner import run
        run(config, root, resume=args.resume)


if __name__ == "__main__":
    main()
