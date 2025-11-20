from dataclasses import dataclass

@dataclass
class TrainConfig:
    batch_size: int = 128
    epochs: int = 20
    lr: float = 1e-3
    num_workers: int = 4
    model_name: str = "basic_cnn"
    device: str = "cuda"  # "cuda" or "cpu"

@dataclass
class ModeConfig:
    mode: str = "local"  # "local" | "gcp-test" | "gcp-prod"
