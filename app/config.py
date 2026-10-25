from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_dir: Path = Path("models/artifacts")
    mode: str = "immediate"
    batch_size: int = 8
    queue_delay_ms: int = 10
    queue_capacity: int = 8
    request_timeout_ms: int = 5000
    shutdown_timeout_ms: int = 5000
    intra_op_threads: int = 2
    inter_op_threads: int = 1
    max_requests: int = 8
    preprocessing_workers: int = 2
    max_upload_bytes: int = 5 * 1024 * 1024
    max_image_pixels: int = 16_000_000
    max_image_side: int = 8192

    def __post_init__(self):
        if self.mode not in {"immediate", "fixed", "dynamic"} or self.batch_size > 8:
            raise ValueError("Invalid mode or batch size")
        for name in self.__dataclass_fields__:
            if name not in {"model_dir", "mode"} and getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")

    @classmethod
    def from_env(cls):
        defaults = cls()
        values = {name: int(os.getenv(f"INFERENCEOPS_{name.upper()}", getattr(defaults, name)))
                  for name in cls.__dataclass_fields__ if name not in {"model_dir", "mode"}}
        return cls(mode=os.getenv("INFERENCEOPS_MODE", defaults.mode), model_dir=Path(os.getenv("INFERENCEOPS_MODEL_DIR", defaults.model_dir)), **values)
