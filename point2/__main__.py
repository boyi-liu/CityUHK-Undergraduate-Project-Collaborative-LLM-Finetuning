"""CLI: python -m point2 prepare|run --config configs/point2-smoke.yaml."""

import argparse
import os
from pathlib import Path

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Project task 2 federated LoRA pipeline")
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--config", default="configs/point2-smoke.yaml")
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
    else:
        if args.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        from .runner import run
        run(config, root, resume=args.resume)


if __name__ == "__main__":
    main()
