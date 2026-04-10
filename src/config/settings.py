"""Project configuration primitives."""

from dataclasses import dataclass

DEFAULT_LOG_LEVEL = "INFO"


@dataclass(frozen=True)
class SplitConfig:
    """Container for reproducible dataset split ratios."""

    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2

    def validate(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                "Split ratios must sum to 1.0, "
                f"received {self.train_ratio}, {self.val_ratio}, {self.test_ratio}."
            )


DEFAULT_SPLIT_CONFIG = SplitConfig()
