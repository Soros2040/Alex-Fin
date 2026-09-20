from dataclasses import dataclass


@dataclass
class ModelConfig:
    d_model: int = 128
    num_heads: int = 4
    num_experts: int = 8
    top_k: int = 2


@dataclass
class TrainingConfig:
    epochs: int = 100
    group_size: int = 16
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    train_years: float = 2.0
    val_years: float = 1.0
    test_years: float = 0.5
    lookback: int = 20
