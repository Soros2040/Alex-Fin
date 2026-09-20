import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="AlexFin端到端工件烟雾校验")
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--require-epoch90", action="store_true")
    args = parser.parse_args()
    model_id = args.model_id
    checks = {
        "best_json": Path(f"checkpoints/{model_id}_best.json").exists(),
        "last_json": Path(f"checkpoints/{model_id}_last.json").exists(),
        "best_weights": Path(f"checkpoints/{model_id}_best_model.pt").exists(),
        "last_weights": Path(f"checkpoints/{model_id}_last_model.pt").exists(),
        "metrics_summary": Path(f"artifacts/metrics/{model_id}_metrics_summary.json").exists(),
        "eval_report": Path(f"artifacts/reports/{model_id}_evaluation_report.json").exists(),
        "eval_mvo": Path(f"artifacts/reports/{model_id}_eval_mvo_comparison.json").exists(),
        "eval_replay": Path(f"artifacts/reports/{model_id}_eval_replay_audit.json").exists(),
        "train_mvo": Path(f"artifacts/reports/{model_id}_mvo_comparison.json").exists(),
        "final_replay": Path(f"artifacts/reports/{model_id}_final_replay_audit.json").exists(),
        "final_trades": Path(f"artifacts/trades/{model_id}_final_test_trades.csv").exists(),
    }
    if args.require_epoch90:
        checks["epoch90_trades"] = Path(f"artifacts/trades/{model_id}_epoch90_test_trades.csv").exists()
        checks["epoch90_summary"] = Path(f"artifacts/trades/{model_id}_epoch90_test_summary.json").exists()
        checks["epoch90_replay"] = Path(f"artifacts/reports/{model_id}_epoch90_replay_audit.json").exists()
    passed = all(checks.values())
    print(json.dumps({"passed": passed, "model_id": model_id, "checks": checks}, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
