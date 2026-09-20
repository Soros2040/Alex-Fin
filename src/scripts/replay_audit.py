import argparse
import json

from src.training.replay_audit import replay_trade_log


def main() -> None:
    parser = argparse.ArgumentParser(description="交易明细回放审计")
    parser.add_argument("--trade-csv", type=str, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    result = replay_trade_log(args.trade_csv, tolerance=args.tolerance)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
