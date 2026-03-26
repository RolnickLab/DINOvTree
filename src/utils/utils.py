import importlib
import logging
import os
import random
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import CSVLogger, WandbLogger

DEFAULT_SEED: Final[int] = 42
CONFIG_DIR: Final[Path] = Path("config")
_BASE: Final[str] = "src.models"
MODEL_REGISTRY: Final[dict[str, dict[str, str]]] = {
    "allometric": {
        "mod": f"{_BASE}.allometric_model.allometric_model",
        "cls": "AllometricModel",
    },
    "resnet": {
        "mod": f"{_BASE}.resnet_model.resnet_model",
        "cls": "ResNetModel",
    },
    "dinov3": {
        "mod": f"{_BASE}.dinov3_model.dinov3_model",
        "cls": "DINOv3Model",
    },
    "dinovtree": {
        "mod": f"{_BASE}.dinovtree_model.dinovtree_model",
        "cls": "DINOvTreeModel",
    },
    "maskrcnn": {
        "mod": f"{_BASE}.maskrcnn_model.maskrcnn_model",
        "cls": "MaskRCNNModel",
    },
}


def prepare_config(cfg: DictConfig, mode: str) -> DictConfig:
    """
    Resolves, flattens, and persists the experiment configuration.

    Args:
        cfg: The raw OmegaConf configuration object.
        mode: Execution mode (e.g., 'train', 'test') for filename labeling.

    Returns:
        The resolved and flattened DictConfig.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(cfg, CONFIG_DIR / f"{mode}_config_unresolved.yaml")

    resolved_cfg = cfg.copy()
    OmegaConf.set_struct(resolved_cfg, False)

    merged = OmegaConf.merge(resolved_cfg, resolved_cfg.model, resolved_cfg.dataset)
    resolved_cfg = cast(DictConfig, merged)

    del resolved_cfg.model
    del resolved_cfg.dataset

    exclude_classes = resolved_cfg.get("exclude_classes") or []
    resolved_cfg.n_classes = resolved_cfg.n_classes - len(set(exclude_classes))

    resolved_cfg.main_metric_name = _determine_main_metric(resolved_cfg.tasks)

    OmegaConf.save(resolved_cfg, CONFIG_DIR / f"{mode}_config_resolved.yaml")

    return resolved_cfg


def _determine_main_metric(tasks: list[str]) -> str:
    """
    Determine the primary metric name based on active tasks.

    Args:
        tasks: A list of strings representing the active machine learning
            tasks (e.g., ["classification", "height"]).

    Returns:
        A string representing the primary metric key to be used for
        model selection and logging.

    Raises:
        ValueError: If the provided tasks list is empty or contains
            unsupported task combinations.
    """
    if len(tasks) > 1:
        return "cm"

    task_map = {
        "classification": "f1",
        "height": "height_delta25",
        "segmentation": "iou",
    }

    for task, metric in task_map.items():
        if task in tasks:
            return metric

    raise ValueError(f"Unknown main metric for tasks: {tasks}")


def setup_logger(cfg: DictConfig) -> tuple[logging.Logger, WandbLogger | CSVLogger]:
    """
    Configures the console logger and the Lightning experiment logger.

    Args:
        cfg: The resolved configuration object.

    Returns:
        A tuple of (standard_logger, pl_logger).
    """
    console_logger = logging.getLogger(__name__)
    if not console_logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    if cfg.get("debug", False):
        pl_logger = CSVLogger(save_dir="logs", name="debug")
    else:
        config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        pl_logger = WandbLogger(
            name=cfg.exp_name,
            project=str(cfg.model_name).replace("_", "-"),
            config=config_dict,
            group=cfg.get("group"),
        )

    return console_logger, pl_logger


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    """
    Sets seeds for reproducibility across Python, NumPy, PyTorch, and Lightning.

    Args:
        seed: The integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    os.environ["PYTHONHASHSEED"] = str(seed)
    pl.seed_everything(seed, workers=True)


class SingleLRMonitor(pl.Callback):
    """Monitors and logs the learning rate of a specific parameter group."""

    def __init__(self, optimizer_idx: int = 0, param_group_idx: int = 0, name: str = "lr") -> None:
        """
        Initialize the monitor with specific optimizer and group indices.

        Args:
            optimizer_idx: Index of the optimizer to monitor.
            param_group_idx: Index of the parameter group in the optimizer to log.
            name: The metric name used in the logs.
        """
        super().__init__()
        self.optimizer_idx = optimizer_idx
        self.param_group_idx = param_group_idx
        self.name = name

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """
        Logs the learning rate at the end of every training batch.

        Args:
            trainer: The PyTorch Lightning Trainer instance.
            pl_module: The LightningModule being trained.
            outputs: The outputs from the training step.
            batch: The current batch of data.
            batch_idx: The index of the current batch within the epoch.
        """
        optimizer = trainer.optimizers[self.optimizer_idx]
        lr = optimizer.param_groups[self.param_group_idx]["lr"]

        pl_module.log(
            self.name,
            lr,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            batch_size=getattr(pl_module, "batch_size", None),
        )


def create_model(cfg: DictConfig) -> pl.LightningModule:
    """
    Factory function to create a model based on the configuration.

    Args:
        cfg: Configuration object containing 'model_name'.

    Returns:
        An instance of the requested LightningModule.

    Raises:
        ValueError: If the model_name is not found in the registry.
    """
    model_name = cfg.model_name

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}. " f"Available models: {list(MODEL_REGISTRY.keys())}")

    entry = MODEL_REGISTRY[model_name]

    module = importlib.import_module(entry["mod"])
    model_class = getattr(module, entry["cls"])

    return model_class(cfg)


def collate_fn(
    batch: list[tuple[Any, np.ndarray, Any, np.ndarray | None, dict[str, Any]]],
) -> tuple[list[Any], torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
    """
    Custom collate function for multi-modal geospatial data.

    Args:
        batch: A list of samples where each sample is a tuple of
            (meta, image, points, dsm, targets_dict).

    Returns:
        A tuple containing:
            - metas: List of metadata objects.
            - images: Stacked float tensor [B, C, H, W].
            - dsms: Stacked float tensor [B, 1, H, W] or None.
            - targets: Dictionary of tensors for various tasks.
    """
    metas, images_np, _, dsms_np, target_dicts = zip(*batch)

    images = torch.from_numpy(np.stack(images_np)).float()

    dsms = None
    if dsms_np[0] is not None:
        dsms = torch.from_numpy(np.stack(dsms_np)).float()

    targets: dict[str, torch.Tensor] = {}
    first_target = target_dicts[0]

    if "class" in first_target:
        classes = [t["class"] - 1 for t in target_dicts]
        targets["class"] = torch.as_tensor(classes, dtype=torch.long)

    if "height" in first_target:
        heights = [float(t["height"]) for t in target_dicts]
        targets["height"] = torch.as_tensor(heights, dtype=torch.float32)

    if "segmentation" in first_target:
        masks = np.stack([t["segmentation"] for t in target_dicts])
        targets["segmentation"] = torch.from_numpy(masks).float()

    return list(metas), images, dsms, targets
