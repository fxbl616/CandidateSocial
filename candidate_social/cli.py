"""Command-line evaluation and prediction for the released checkpoints."""
import argparse
import json

import torch

from .inference import evaluate_checkpoint, predict_checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(prog="candidate_social")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("evaluate", "predict"):
        command = commands.add_parser(name)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:0")
        command.add_argument("--batch-size", type=int, default=4)
        command.add_argument("--num-threads", type=int, default=4)
        if name == "evaluate":
            command.add_argument("--data-root", required=True)
            command.add_argument("--num-workers", type=int, default=0)
            command.add_argument("--max-windows", type=int, default=0,
                                 help="Smoke check on a prefix; 0 evaluates the complete test set")
        else:
            command.add_argument("--input", required=True, help="NPZ with obs and agent_mask")
    args = parser.parse_args(argv)
    if args.num_threads < 1:
        parser.error("--num-threads must be positive")
    torch.set_num_threads(args.num_threads)
    try:
        if args.command == "evaluate":
            result = evaluate_checkpoint(args.checkpoint, args.data_root, args.output,
                                         device=args.device, batch_size=args.batch_size,
                                         num_workers=args.num_workers, max_windows=args.max_windows)
            print(json.dumps({"output": args.output, "metrics": result["metrics"]}, indent=2))
        else:
            result = predict_checkpoint(args.checkpoint, args.input, args.output,
                                        device=args.device, batch_size=args.batch_size)
            print(json.dumps(result, indent=2))
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        parser.error(str(error))
    return 0
