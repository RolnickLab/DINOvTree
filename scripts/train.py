import logging
from pathlib import Path
from typing import Final

import hydra
import pytorch_lightning as pl
import torch
from geodataset.dataset.multimodal_dataset import (
    LabeledMultiModalCocoDataset,
)
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Subset

from src.utils.augmentor import ImageAugmentor
from src.utils.utils import (
    SingleLRMonitor,
    collate_fn,
    create_model,
    prepare_config,
    seed_everything,
    setup_logger,
)

DEBUG_SUBSET_STEP: Final[int] = 10


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for the training pipeline.

    Args:
        cfg: Hydra-provided DictConfig containing experiment parameters.
    """
    cfg = prepare_config(cfg, mode="train")
    console_logger, pl_logger = setup_logger(cfg)
    seed_everything(cfg.seed)

    console_logger.info(f"Running training in directory: {Path.cwd()}")
    console_logger.info(f"Running with config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    callbacks = _setup_callbacks(cfg)
    train_loader, val_loader, test_loader = _setup_dataloaders(cfg, console_logger)
    model = create_model(cfg)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1 if torch.cuda.is_available() else 1,
        logger=pl_logger,
        callbacks=callbacks,
        max_epochs=cfg.max_epochs,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        enable_progress_bar=True,
        deterministic=True,
        profiler="simple",
        accumulate_grad_batches=cfg.accumulate_grad_batches,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    ckpt_path = None if cfg.debug else "best"
    trainer.test(model=model, dataloaders=test_loader, ckpt_path=ckpt_path)


def _setup_callbacks(cfg: DictConfig) -> list[pl.Callback]:
    """
    Configures Lightning callbacks based on debug mode.

    Args:
        cfg: Configuration object.

    Returns:
        A list of initialized PyTorch Lightning callbacks.
    """
    if cfg.debug:
        return []

    checkpoint_dir = Path.cwd().parent / "checkpoints"

    best_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=(f"best-{cfg.exp_name}-{{epoch:02d}}-" f"{{val/{cfg.main_metric_name}:.4f}}"),
        monitor=f"val/{cfg.main_metric_name}",
        mode="max",
        save_top_k=1,
    )

    last_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"last-{cfg.exp_name}-{{epoch:02d}}",
        save_last=False,
    )

    lr_monitor = SingleLRMonitor(optimizer_idx=0, param_group_idx=-1)

    return [best_cb, last_cb, lr_monitor]


def _setup_dataloaders(cfg: DictConfig, console_logger: logging.Logger) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Handles the configuration of augmentations, datasets, and dataloaders.

    Args:
        cfg: The experiment configuration.
        console_logger: Logger for reporting dataset statistics.

    Returns:
        A tuple containing (train_loader, val_loader, test_loader).
    """
    augmentor = ImageAugmentor(geo_p=cfg.augmentations.geo_p) if cfg.data_aug else None

    folds = ["train", "valid", "test"]
    datasets = {}

    for fold in folds:
        current_aug = augmentor if fold == "train" else None

        ds = LabeledMultiModalCocoDataset(
            dataset_name=cfg.dataset_name,
            fold=fold,
            modalities=cfg.modalities,
            tasks=cfg.tasks,
            root_path=cfg.data_path,
            augment=current_aug,
            exclude_classes=cfg.exclude_classes,
            height_attr=cfg.height_attr,
        )

        if cfg.debug:
            indices = list(range(0, len(ds), DEBUG_SUBSET_STEP))
            ds = Subset(ds, indices=indices)

        datasets[fold] = ds

    console_logger.info(
        f"Samples - Train: {len(datasets['train'])}, "
        f"Val: {len(datasets['valid'])}, "
        f"Test: {len(datasets['test'])}"
    )

    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "collate_fn": collate_fn,
        "num_workers": cfg.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(datasets["train"], shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(datasets["valid"], shuffle=False, **loader_kwargs)
    test_loader = DataLoader(datasets["test"], shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    main()
