import argparse
import json

from src.training.trainer import TrainConfig, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="AlexFin 训练入口")
    parser.add_argument("--model-id", type=str, default="alexfin_base")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--top-n-stocks", type=int, default=50)
    parser.add_argument("--data-path", type=str, default="src/data/cache/daily.parquet")
    parser.add_argument("--lightweight-data", action="store_true", help="使用轻量化数据")
    parser.add_argument("--lightweight-data-dir", type=str, default="src/data-lightweight/data", help="轻量化数据目录")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    parser.add_argument("--train-years", type=float, default=7.0)
    parser.add_argument("--val-years", type=float, default=1.0)
    parser.add_argument("--test-years", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-checkpoint", type=str, default=None)
    parser.add_argument("--base-model-id", type=str, default=None)
    parser.add_argument("--data-version", type=str, default="daily_cache_v1")
    parser.add_argument("--feature-version", type=str, default="feature_mask_v1")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k-experts", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=1e-3)
    parser.add_argument("--rolling-step-years", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--glasso-alpha", type=float, default=0.02)
    parser.add_argument("--glasso-min-history", type=int, default=60)
    parser.add_argument("--require-glasso-dy", action="store_true")
    parser.add_argument("--quick-mode", action="store_true")
    args = parser.parse_args()

    config = TrainConfig(
        data_path=args.data_path,
        lightweight_data=args.lightweight_data,
        lightweight_data_dir=args.lightweight_data_dir,
        artifacts_dir=args.artifacts_dir,
        checkpoints_dir=args.checkpoints_dir,
        top_n_stocks=args.top_n_stocks,
        epochs=args.epochs,
        train_years=args.train_years,
        val_years=args.val_years,
        test_years=args.test_years,
        seed=args.seed,
        model_id=args.model_id,
        base_checkpoint_path=args.base_checkpoint,
        base_model_id=args.base_model_id,
        data_version=args.data_version,
        feature_version=args.feature_version,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_experts=args.num_experts,
        top_k_experts=args.top_k_experts,
        group_size=args.group_size,
        learning_rate=args.learning_rate,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        rolling_step_years=args.rolling_step_years,
        early_stop_patience=args.early_stop_patience,
        glasso_alpha=args.glasso_alpha,
        glasso_min_history=args.glasso_min_history,
        require_glasso_dy=args.require_glasso_dy,
        quick_mode=args.quick_mode,
    )
    result = run_training(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
