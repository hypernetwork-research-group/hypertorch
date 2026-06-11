import logging

from .latex_logger import LaTexTableConfig, LaTexTableLogger, colorize_metric_value

from .markdown_logger import MarkdownTableConfig, MarkdownTableLogger

from .trainer import MultiModelTrainer

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

__all__ = [
    "LaTexTableConfig",
    "LaTexTableLogger",
    "MarkdownTableConfig",
    "MarkdownTableLogger",
    "MultiModelTrainer",
    "colorize_metric_value",
]
