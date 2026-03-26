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
from torch.utils.data import DataLoader, Subset

from src.utils.utils import (
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
    Main entry point for the model evaluation pipeline.

    Args:
        cfg: Hydra-provided DictConfig containing experiment parameters.
    """
    cfg = prepare_config(cfg, mode="eval")
    console_logger, pl_logger = setup_logger(cfg)
    seed_everything(cfg.seed)

    console_logger.info(f"Running evaluation in directory: {Path.cwd()}")
    console_logger.info(f"Running with config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    test_loader = _setup_test_dataloader(cfg, console_logger)
    model = create_model(cfg)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1 if torch.cuda.is_available() else 1,
        logger=pl_logger,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        deterministic=True,
        profiler="simple",
    )

    model = create_model(cfg)

    ckpt_path: Path | None = None
    if cfg.get("ckpt_path") and isinstance(cfg.ckpt_path, str):
        ckpt_path = Path(hydra.utils.get_original_cwd()) / cfg.ckpt_path
        if ckpt_path.suffix == ".pth":
            console_logger.info(f"Loading weights-only checkpoint: {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=True)
            ckpt_path = None
        elif ckpt_path.suffix == ".ckpt":
            console_logger.info(f"Loading full Lightning checkpoint: {ckpt_path}")
        else:
            raise ValueError(f"Unsupported checkpoint extension: {ckpt_path}. Use .pth or .ckpt")

    trainer.test(model=model, dataloaders=test_loader, ckpt_path=ckpt_path)


def _setup_test_dataloader(cfg: DictConfig, console_logger: logging.Logger) -> DataLoader:
    """
    Configures the dataset and DataLoader for evaluation.

    Args:
        cfg: The experiment configuration.
        console_logger: Logger for reporting dataset statistics.

    Returns:
        A PyTorch DataLoader configured for the test set.
    """
    test_set = LabeledMultiModalCocoDataset(
        dataset_name=cfg.dataset_name,
        fold="test",
        modalities=cfg.modalities,
        tasks=cfg.tasks,
        root_path=cfg.data_path,
        augment=None,
        exclude_classes=cfg.exclude_classes,
        height_attr=cfg.height_attr,
    )

    if cfg.debug and not cfg.get("write_predictions", False):
        indices = list(range(0, len(test_set), DEBUG_SUBSET_STEP))
        test_set = Subset(test_set, indices=indices)
        console_logger.info(f"Debug mode: Using subset of {len(test_set)} samples.")

    console_logger.info(f"Number of test samples: {len(test_set)}")

    return DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


if __name__ == "__main__":
    main()
