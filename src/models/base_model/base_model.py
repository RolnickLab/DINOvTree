import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Final, Sized, Tuple, cast

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import ConfusionMatrixDisplay
from torchmetrics import (
    Accuracy,
    ConfusionMatrix,
    F1Score,
    JaccardIndex,
    MeanAbsoluteError,
    MeanSquaredError,
    Metric,
    MetricCollection,
)
from torchmetrics.regression import MeanSquaredLogError

from src.utils.metrics import (
    ClasswiseRegressionWrapper,
    DeltaAccuracy,
)
from src.utils.scheduler import build_cosine_scheduler


class BaseModel(pl.LightningModule, ABC):
    """
    Abstract base class for all project models using PyTorch Lightning.

    This class handles configuration parsing, category mapping, and metric initialization for classification,
    segmentation, and height regression tasks.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initializes the base model and metric collections.

        Args:
            cfg: Configuration object containing model and dataset parameters.
        """
        super().__init__()
        self.cfg: Final[DictConfig] = cfg
        self.write_predictions: bool = cfg.write_predictions
        self.modalities: list[str] = cfg.modalities

        self.classification: bool = "classification" in cfg.tasks
        self.height: bool = "height" in cfg.tasks
        self.segmentation: bool = "segmentation" in cfg.tasks

        self.category_dict: dict[int, str] = self._setup_categories()

        if self.write_predictions:
            self.test_step_outputs: list[dict[str, list[str] | torch.Tensor]] = []

        self.build_model()
        self._initialize_metrics()

        self.best_val_score: float = -float("inf")
        self.best_flat_results: dict[str, float] = {}

    def _setup_categories(self) -> dict[int, str]:
        """
        Loads and filters categories from a JSON file.

        Returns:
            A dictionary mapping processed class IDs to sanitized names.
        """
        with open(self.cfg.categories_path, "r") as f:
            categories_data = json.load(f)["categories"]

        exclude = self.cfg.exclude_classes
        if exclude is not None:
            skipped_classes = sorted(set(exclude))
            filtered = [c for c in categories_data if c["id"] not in skipped_classes]

            remap = {item["id"]: i for i, item in enumerate(sorted(filtered, key=lambda x: x["id"]))}
            return {remap[c["id"]]: c["name"].replace(" ", "_").replace(".", "") for c in filtered}

        return {c["id"] - 1: c["name"].replace(" ", "_").replace(".", "") for c in categories_data}

    def _initialize_metrics(self) -> None:
        """Initializes MetricCollections for train, val, and test stages."""
        train_m, val_m, test_m = {}, {}, {}

        if self.classification:
            train_m["classification"] = MetricCollection(
                {"f1": F1Score(task="multiclass", num_classes=self.cfg.n_classes, average="macro")}
            )
            val_m["classification"] = self._build_classification_metrics(confmat=False)
            test_m["classification"] = self._build_classification_metrics(confmat=True)
        if self.segmentation:
            train_m["segmentation"] = MetricCollection({"iou": JaccardIndex(task="binary")})
            val_m["segmentation"] = self._build_segmentation_metrics()
            test_m["segmentation"] = self._build_segmentation_metrics()
        if self.height:
            train_m["height"] = MetricCollection(
                {
                    "height_delta25": DeltaAccuracy(delta=1.25),
                }
            )
            val_m["height"] = self._build_height_metrics(per_class=False)
            test_m["height"] = self._build_height_metrics(per_class=False)
            if self.classification:
                val_m["height_per_class"] = self._build_height_metrics(per_class=True, category_dict=self.category_dict)
                test_m["height_per_class"] = self._build_height_metrics(
                    per_class=True, category_dict=self.category_dict
                )

        self.metrics: nn.ModuleDict = nn.ModuleDict(
            {
                "train_metrics": nn.ModuleDict(train_m),
                "val_metrics": nn.ModuleDict(val_m),
                "test_metrics": nn.ModuleDict(test_m),
            }
        )
        self.media_tables = {}

    def _build_classification_metrics(self, confmat: bool) -> MetricCollection:
        """
        Builds a collection of metrics for multiclass classification.

        Args:
            confmat: Whether to include a confusion matrix in the collection.

        Returns:
            A MetricCollection containing macro and per-class Accuracy and F1.
        """
        metrics: dict[str, Metric | MetricCollection] = {
            "acc": Accuracy(task="multiclass", num_classes=self.cfg.n_classes, average="macro"),
            "f1": F1Score(task="multiclass", num_classes=self.cfg.n_classes, average="macro"),
            # per-class
            "acc_per_class": Accuracy(task="multiclass", num_classes=self.cfg.n_classes, average=None),
            "f1_per_class": F1Score(task="multiclass", num_classes=self.cfg.n_classes, average=None),
        }
        if confmat:
            metrics["confmat"] = ConfusionMatrix(task="multiclass", num_classes=self.cfg.n_classes)
        return MetricCollection(metrics)

    def _build_segmentation_metrics(self) -> MetricCollection:
        """
        Builds a collection of metrics for binary segmentation tasks.

        Returns:
            A MetricCollection containing the Jaccard Index (IoU).
        """
        return MetricCollection(
            {
                "iou": JaccardIndex(task="binary"),
            },
            compute_groups=False,
        )

    def _build_height_metrics(
        self, per_class: bool = False, category_dict: dict[int, str] | None = None
    ) -> MetricCollection | ClasswiseRegressionWrapper:
        """
        Builds metrics for height regression, optionally wrapped for class-wise analysis.

        Args:
            per_class: If True, wraps metrics to calculate them separately per category.
            category_dict: Mapping of class IDs to names, required if per_class is True.

        Returns:
            A MetricCollection or a ClasswiseRegressionWrapper containing MAE,
            RMSE, MSLE, and DeltaAccuracy.
        """
        base_metrics = MetricCollection(
            {
                "height_mae": MeanAbsoluteError(),
                "height_rmse": MeanSquaredError(squared=False),
                "height_msle": MeanSquaredLogError(),
                "height_delta25": DeltaAccuracy(delta=1.25),
            },
            compute_groups=False,
        )
        if per_class and category_dict is not None:
            return ClasswiseRegressionWrapper(base_metrics, category_dict)
        else:
            return base_metrics

    @abstractmethod
    def build_model(self) -> None:
        """Abstract method to initialize model architecture."""
        raise NotImplementedError("Subclasses must implement build_model.")

    @abstractmethod
    def forward(
        self, batch: tuple[list[Any], torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]
    ) -> tuple[dict[str, torch.Tensor | None], torch.Tensor]:
        """Abstract method for the model forward pass."""
        raise NotImplementedError("Subclasses must implement forward.")

    @abstractmethod
    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict[str, Any]]]:
        """Abstract method to configure optimizers."""
        raise NotImplementedError("Subclasses must implement configure_optimizers.")

    def build_cosine_scheduler(self, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
        """
        Builds a cosine learning rate scheduler with warmup.

        Args:
            optimizer: The optimizer for which to build the scheduler.

        Returns:
            A dictionary containing the scheduler and its configuration metadata
            compatible with PyTorch Lightning.
        """
        total_steps: int = int(self.trainer.estimated_stepping_batches)
        niter_per_epoch = total_steps // self.cfg.max_epochs
        scheduler = build_cosine_scheduler(
            optimizer=optimizer,
            base_lr=self.cfg.lr_head,
            total_steps=total_steps,
            warmup_epochs=self.cfg.warmup_epochs,
            niter_per_epoch=niter_per_epoch,
            warmup_lr=self.cfg.warmup_lr,
            min_lr=self.cfg.min_lr,
        )

        return scheduler

    def training_step(
        self, batch: tuple[list[Any], torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]], batch_idx: int
    ) -> torch.Tensor:
        """
        Performs a single training step.

        Args:
            batch: Tuple of (metadata, images, dsms, targets).
            batch_idx: Index of the current batch.

        Returns:
            The calculated loss tensor.
        """
        pred, loss = self(batch)
        self._update_metrics_and_visuals("train", batch_idx, batch, pred, loss)
        return loss

    def validation_step(
        self, batch: tuple[list[Any], torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]], batch_idx: int
    ) -> torch.Tensor:
        """
        Performs a single validation step.

        Args:
            batch: Tuple of (metadata, images, dsms, targets).
            batch_idx: Index of the current batch.

        Returns:
            The calculated loss tensor.
        """
        pred, loss = self(batch)
        self._update_metrics_and_visuals("val", batch_idx, batch, pred, loss)
        return loss

    def test_step(
        self, batch: tuple[list[Any], torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]], batch_idx: int
    ) -> torch.Tensor:
        """
        Performs a single test step and optionally logs predictions.

        Args:
            batch: Tuple of (metadata, images, dsms, targets).
            batch_idx: Index of the current batch.

        Returns:
            The calculated loss tensor.
        """
        pred, loss = self(batch)
        self._update_metrics_and_visuals("test", batch_idx, batch, pred, loss)
        if self.write_predictions:
            meta, _, _, targets = batch

            output_data: dict[str, torch.Tensor | list[str]] = {"filenames": [m["filename"] for m in meta]}

            if self.classification:
                output_data["class_preds"] = pred["class"].detach().cpu()
                output_data["class_targets"] = targets["class"].detach().cpu()

            if self.height:
                output_data["height_preds"] = pred["height"].detach().cpu()
                output_data["height_targets"] = targets["height"].detach().cpu()

            self.test_step_outputs.append(output_data)

        return loss

    def _setup_media_log_batches(self, stage: str, num_logs: int) -> list[int]:
        """
        Calculates batch indices for logging media based on stage length.

        Args:
            stage: The stage name ('train', 'val', or 'test').
            num_logs: Number of media samples to log per epoch.

        Returns:
            A list of batch indices at which to trigger media logging.
        """
        suffix = "s" if stage != "train" else ""
        loader = getattr(self.trainer, f"{stage}_dataloader{suffix}", None)
        target = loader[0] if isinstance(loader, list) and loader else loader
        total_batches = len(cast(Sized, target)) if target else 0
        return [total_batches * i // num_logs for i in range(num_logs)]

    def on_train_start(self) -> None:
        """Sets up training media logging indices."""
        self.train_media_log_batches = self._setup_media_log_batches("train", self.cfg.num_train_media_logs)

    def on_validation_start(self) -> None:
        """Sets up validation media logging indices."""
        self.val_media_log_batches = self._setup_media_log_batches("val", self.cfg.num_val_media_logs)

    def on_test_start(self) -> None:
        """Sets up test media logging indices."""
        self.test_media_log_batches = self._setup_media_log_batches("test", self.cfg.num_test_media_logs)

    def on_train_epoch_end(self) -> None:
        """Computes and logs training metrics, then resets them."""
        results: dict[str, torch.Tensor] = {}
        train_metrics = cast(nn.ModuleDict, self.metrics["train_metrics"])
        for task in self.cfg.tasks:
            task_metrics = cast(MetricCollection, train_metrics[task])
            results.update(task_metrics.compute())
            task_metrics.reset()

        results = self.add_combined_metric(results)
        self._log_metrics(results, "train")

        criterion = getattr(self, "criterion", None)
        if criterion is not None and hasattr(criterion, "step_epoch"):
            criterion.step_epoch()

    def on_validation_epoch_end(self) -> None:
        """Logs validation metrics and tracks the best performing score."""
        results: dict[str, torch.Tensor] = {}
        val_metrics = cast(nn.ModuleDict, self.metrics["val_metrics"])
        for task in self.cfg.tasks:
            task_metrics = cast(MetricCollection, val_metrics[task])
            results.update(task_metrics.compute())
            task_metrics.reset()

        results = self.compute_height_metrics_per_class(results, metric_prefix="val")
        results = self.add_combined_metric(results)
        flat_results = self._log_metrics(results, "val")

        current_score = flat_results[f"val/{self.cfg.main_metric_name}"]
        if current_score > self.best_val_score:
            self.best_val_score = current_score
            self.best_flat_results = flat_results.copy()

    def on_test_epoch_end(self) -> None:
        """Finalizes test metrics and saves predictions to disk if configured."""
        results: dict[str, torch.Tensor] = {}
        test_metrics = cast(nn.ModuleDict, self.metrics["test_metrics"])
        for task in self.cfg.tasks:
            task_metrics = cast(MetricCollection, test_metrics[task])
            results.update(task_metrics.compute())
            task_metrics.reset()

        results = self.compute_height_metrics_per_class(results, metric_prefix="test")
        results = self.add_combined_metric(results)
        flat_results = self._log_metrics(results, "test")

        self.best_flat_results.update(flat_results)

        if not self.cfg.debug:
            wandb_logger = cast(WandbLogger, self.logger)
            for k, v in self.best_flat_results.items():
                wandb_logger.experiment.summary[k] = v

        if self.write_predictions:
            self._save_test_predictions()

    def _save_test_predictions(self) -> None:
        """Aggregates test step outputs and saves them to a CSV file."""
        all_filenames = [fname for batch_out in self.test_step_outputs for fname in batch_out["filenames"]]
        data_to_save: dict[str, Any] = {
            "filename": all_filenames,
        }

        def _get_concatenated_array(key: str) -> Any:
            tensors: list[torch.Tensor] = [
                val for b in self.test_step_outputs if (val := b.get(key)) is not None and isinstance(val, torch.Tensor)
            ]
            return torch.cat(tensors, dim=0).cpu().numpy().flatten()

        if self.classification:
            data_to_save["class_pred"] = _get_concatenated_array("class_preds")
            data_to_save["class_target"] = _get_concatenated_array("class_targets")

        if self.height:
            data_to_save["height_pred"] = _get_concatenated_array("height_preds")
            data_to_save["height_target"] = _get_concatenated_array("height_targets")

        df = pd.DataFrame(data_to_save)
        df.to_csv(self.cfg.test_results_path, index=False)
        self.test_step_outputs.clear()

        print(f"Saved {len(df)} test predictions to {self.cfg.test_results_path}")

    def compute_height_metrics_per_class(
        self, results_dict: dict[str, torch.Tensor], metric_prefix: str = "val"
    ) -> dict[str, torch.Tensor]:
        """
        Computes per-class height metrics and updates the results dictionary.

        This method extracts the class-wise regression metrics from the stage's
        ModuleDict, computes the values, and resets the metric states.

        Args:
            results_dict: The dictionary of results to be updated.
            metric_prefix: The current stage prefix (e.g., 'val' or 'test').

        Returns:
            The updated results_dict containing flattened per-class metrics.
        """
        stage_metrics = cast(nn.ModuleDict, self.metrics[f"{metric_prefix}_metrics"])
        if "height_per_class" in stage_metrics:
            height_wrapper = cast(ClasswiseRegressionWrapper, stage_metrics["height_per_class"])
            results_dict.update(height_wrapper.compute())
            height_wrapper.reset()
        return results_dict

    def add_combined_metric(self, results_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Calculates an unweighted average of the primary metrics for all active tasks.

        Args:
            results_dict: Dictionary containing the computed metrics for the current epoch.

        Returns:
            The results_dict updated with the 'cm' (combined metric) key.
        """
        device = next(iter(results_dict.values())).device
        cm = torch.tensor(0.0, device=device)
        if self.classification:
            cm += results_dict["f1"]
        if self.segmentation:
            cm += results_dict["iou"]
        if self.height:
            cm += results_dict["height_delta25"]
        cm = cm / len(self.cfg.tasks)
        results_dict["cm"] = cm
        return results_dict

    def _log_metrics(self, metrics_dict: dict[str, torch.Tensor | Any], stage: str) -> dict[str, float]:
        """
        Flattens metrics, handles confusion matrix visualization, and logs to the logger.

        Args:
            metrics_dict: Raw metrics from MetricCollections or Wrappers.
            stage: Current execution stage ('train', 'val', or 'test').

        Returns:
            A flattened dictionary of float values for historical tracking.
        """
        flat_results: dict[str, float] = {}
        wandb_logs: dict[str, float] = {}

        for k, v in metrics_dict.items():
            if k == "confmat":
                self._handle_confusion_matrix(v, stage)

            elif isinstance(v, torch.Tensor) and v.numel() > 1:
                for i, val in enumerate(v):
                    cls_name = f"{stage}/{k}_{self.category_dict[i]}"
                    item_val = val.item()
                    flat_results[cls_name] = item_val
                    wandb_logs[cls_name] = item_val

            else:
                name = f"{stage}/{k}"
                item_val = v.item()
                flat_results[name] = item_val
                if "height_" not in k or "per_class" not in k:
                    wandb_logs[name] = item_val

        self.log_dict(wandb_logs, on_step=False, prog_bar=False, batch_size=self.cfg.batch_size)
        return flat_results

    def _handle_confusion_matrix(self, confmat_tensor: torch.Tensor, stage: str) -> None:
        """
        Processes, saves, and logs the confusion matrix visualization.

        This method converts the confusion matrix tensor to a NumPy array, persists it
        as a CSV file, generates a matplotlib figure, and logs the image to Weights & Biases.

        Args:
            confmat_tensor: Square tensor of shape (num_classes, num_classes)
                containing the confusion matrix data.
            stage: The current execution stage (e.g., 'val' or 'test') used for
                file naming and logging keys.
        """
        confmat = confmat_tensor.cpu().numpy()
        class_names = [self.category_dict[i] for i in range(len(self.category_dict))]

        df = pd.DataFrame(confmat, index=class_names, columns=class_names)
        save_dir = Path("confmat")
        save_dir.mkdir(exist_ok=True)
        df.to_csv(save_dir / f"{stage}_cm_epoch{self.current_epoch}.csv")

        fig, ax = plt.subplots(figsize=(10, 10))
        disp = ConfusionMatrixDisplay(confusion_matrix=confmat, display_labels=class_names)
        disp.plot(include_values=True, cmap="Blues", ax=ax, colorbar=False, xticks_rotation=90)
        ax.set_title(f"{stage.capitalize()} Confusion Matrix", fontsize=14)
        plt.tight_layout()

        if not self.cfg.debug and isinstance(self.logger, WandbLogger):
            self.logger.experiment.log({f"{stage}/confusion_matrix": wandb.Image(fig)})

        fig.savefig(save_dir / f"{stage}_confusion_matrix_epoch{self.current_epoch}.png")
        plt.close(fig)

    def on_train_end(self) -> None:
        """Finalizes the experiment by writing the best results to the logger summary."""
        if not self.cfg.debug and self.best_flat_results and isinstance(self.logger, WandbLogger):
            for k, v in self.best_flat_results.items():
                self.logger.experiment.summary[k] = v

    def _update_metrics_and_visuals(
        self,
        stage: str,
        batch_idx: int,
        batch: tuple[list[Any], torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]],
        pred: dict[str, torch.Tensor],
        loss: torch.Tensor,
    ) -> None:
        """
        Updates stage-specific metrics and handles periodic media logging.

        This method logs the primary loss, updates task-specific metric collections
        (classification, segmentation, height), and triggers rich media logging
        at pre-defined batch intervals.

        Args:
            stage: The current execution phase ('train', 'val', or 'test').
            batch_idx: The index of the current batch within the epoch.
            batch: A tuple containing (metadata, images, optional DSMs, targets).
            pred: Dictionary of model predictions for each active task.
            loss: The computed scalar loss for the current step.
        """
        meta, image, dsm, targets = batch
        self.log(
            f"{stage}/loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            batch_size=self.cfg.batch_size,
        )

        stage_metrics = cast(nn.ModuleDict, self.metrics[f"{stage}_metrics"])

        if self.classification:
            classification_metrics = cast(MetricCollection, stage_metrics["classification"])
            classification_metrics.update(pred["class"], targets["class"])

        if self.segmentation:
            mask_pred = pred["segmentation"]
            mask_true = targets["segmentation"]
            segmentation_metrics = cast(MetricCollection, stage_metrics["segmentation"])
            segmentation_metrics["iou"].update(mask_pred, mask_true)

        if self.height:
            valid_mask = ~torch.isnan(targets["height"])
            if valid_mask.any():
                valid_preds = pred["height"][valid_mask]
                valid_targets = targets["height"][valid_mask]
                height_metrics = cast(MetricCollection, stage_metrics["height"])
                height_metrics.update(valid_preds, valid_targets)
                if "height_per_class" in stage_metrics:
                    valid_labels = targets["class"][valid_mask]
                    height_wrapper = cast(ClasswiseRegressionWrapper, stage_metrics["height_per_class"])
                    height_wrapper.update(valid_preds, valid_targets, valid_labels)

        if not self.cfg.debug:
            log_batches = getattr(self, f"{stage}_media_log_batches", [])

            if batch_idx in log_batches:
                sample_idx: int = 0

                self._log_media_table(
                    stage=stage,
                    image=image[sample_idx],
                    dsm=dsm[sample_idx] if "dsm" in self.modalities and dsm is not None else None,
                    id_true=targets["class"][sample_idx] if self.classification else None,
                    id_pred=pred["class"][sample_idx] if self.classification else None,
                    mask_true=targets["segmentation"][sample_idx] if self.segmentation else None,
                    mask_pred=pred["segmentation"][sample_idx] if self.segmentation else None,
                    resolution=meta[sample_idx]["resolution"],
                    height_true=targets["height"][sample_idx] if self.height else None,
                    height_pred=pred["height"][sample_idx] if self.height else None,
                    filename=meta[sample_idx]["filename"],
                )

    def _log_media_table(
        self,
        stage: str,
        image: torch.Tensor,
        dsm: torch.Tensor | None,
        id_true: torch.Tensor | None,
        id_pred: torch.Tensor | None,
        mask_true: torch.Tensor | None,
        mask_pred: torch.Tensor | None,
        resolution: Tuple[float, float],
        height_true: torch.Tensor | None,
        height_pred: torch.Tensor | None,
        filename: Path,
    ) -> None:
        """
        Assembles and logs an incremental WandB table for multi-modal model evaluation.

        Args:
            stage: Current stage ('train', 'val', or 'test').
            image: Original input image tensor of shape (C, H, W).
            dsm: Digital Surface Model tensor or None.
            id_true: Ground truth class index.
            id_pred: Predicted class index.
            mask_true: Ground truth segmentation mask.
            mask_pred: Predicted segmentation mask.
            resolution: Pixel resolution (x, y) in meters for area calculation.
            height_true: Ground truth height scalar.
            height_pred: Predicted height scalar.
            filename: Path to the source file.
        """
        if stage not in self.media_tables:
            self.media_tables[stage] = self._initialize_empty_table()

        row_data: list[Any] = []

        if "image" in self.modalities:
            if image is not None:
                wandb_img = self._prepare_wandb_image(image, mask_true, mask_pred)
                row_data.append(wandb_img)

        if "dsm" in self.modalities:
            wandb_dsm = None
            if dsm is not None:
                dsm_np = dsm.detach().cpu().permute(1, 2, 0).numpy()
                wandb_dsm = wandb.Image(dsm_np)
            row_data.append(wandb_dsm)

        preds_list, targets_list = [], []

        if self.classification and id_true is not None and id_pred is not None:
            preds_list.append(f"Class: {self.category_dict[int(id_pred.item())]}")
            targets_list.append(f"Class: {self.category_dict[int(id_true.item())]}")

        if self.segmentation and mask_true is not None and mask_pred is not None:
            pixel_area_m2 = resolution[0] * resolution[1]
            area_p = mask_pred.sum().item() * pixel_area_m2
            area_t = mask_true.sum().item() * pixel_area_m2
            preds_list.append(f"Area: {area_p:.1f} (m²)")
            targets_list.append(f"Area: {area_t:.1f} (m²)")

        if self.height and height_true is not None and height_pred is not None:
            preds_list.append(f"Height: {height_pred.item():.2f} (m)")
            targets_list.append(f"Height: {height_true.item():.2f} (m)")

        row_data.append(", ".join(preds_list))
        row_data.append(", ".join(targets_list))
        row_data.append(str(filename) if filename else "N/A")

        self.media_tables[stage].add_data(*row_data)

        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log({f"{stage}/visualizations": self.media_tables[stage]})

    def _initialize_empty_table(self) -> wandb.Table:
        """Defines the column structure for the media table based on active tasks."""
        cols = []
        if "image" in self.modalities:
            cols.append("Image")
        if "dsm" in self.modalities:
            cols.append("DSM")
        targets_column_names = []
        predictions_column_names = []
        if self.classification:
            targets_column_names.append("True Class")
            predictions_column_names.append("Predicted Class")
        if self.segmentation:
            targets_column_names.append("True Area")
            predictions_column_names.append("Predicted Area")
        if self.height:
            targets_column_names.append("True Height")
            predictions_column_names.append("Predicted Height")
        cols.extend([", ".join(predictions_column_names), ", ".join(targets_column_names), "File Name"])
        return wandb.Table(columns=cols, log_mode="INCREMENTAL")

    def _prepare_wandb_image(
        self, image: torch.Tensor, m_true: torch.Tensor | None, m_pred: torch.Tensor | None
    ) -> wandb.Image:
        """
        Prepares a wandb.Image object with optional mask overlays.

        Args:
            image: RGB image tensor (C, H, W).
            m_true: Ground truth mask tensor or None.
            m_pred: Predicted mask tensor or None.

        Returns:
            A wandb.Image object configured with segmentation masks if applicable.
        """
        import wandb

        img_np = image.detach().cpu().permute(1, 2, 0).numpy()

        if self.segmentation and m_true is not None and m_pred is not None:
            return wandb.Image(
                img_np,
                masks={
                    "true": {
                        "mask_data": m_true.detach().cpu().numpy().squeeze(),
                        "class_labels": {1: "Tree of Interest"},
                    },
                    "pred": {
                        "mask_data": m_pred.detach().cpu().numpy().squeeze(),
                        "class_labels": {1: "Tree of Interest"},
                    },
                },
            )
        return wandb.Image(img_np)
