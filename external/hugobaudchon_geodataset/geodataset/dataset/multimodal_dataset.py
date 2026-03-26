import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import rasterio
from geodataset.dataset.base_dataset import BaseLabeledRasterCocoDataset
from geodataset.utils.utils import decode_coco_segmentation
from torch.utils.data import Dataset


class LabeledMultiModalCocoDataset(BaseLabeledRasterCocoDataset, Dataset):
    """
    A dataset class for classification tasks using polygon-based tiles from raster and point cloud data. Loads COCO
    datasets and their associated tiles.

    It can directly be used with a torch.utils.data.DataLoader.

    Parameters
    ----------
    dataset_name: str
        The name identifying the dataset.
    fold: str
        The dataset fold to load ('train', 'valid', 'test').
    root_path: Union[str, List[str], Path, List[Path]]
        The root directory of the dataset
    modalities: List[str]
        List of modalities to include ('image', 'point_cloud', 'dsm').
    tasks: List[str]
        List of tasks to perform ('classification', 'height', 'segmentation').
    voxel_size: Optional[float]
        Voxel size for point cloud downsampling.
    num_points_file: Optional[int]
        Target number of points per file.
    downsample_method_file: Optional[str]
        The method used for file-level downsampling.
    augment: Optional[Any]
        An optional augmentation pipeline for the dataset.
    num_points: Optional[int]
        Fixed number of points to sample during loading.
    few_shot_path: Optional[str]
        Path to few-shot configuration files.
    num_downsample_seeds: int
        Number of seeds to use for downsampling.
    exclude_classes: Optional[List[int]]
        List of COCO category IDs to ignore.
    outlier_removal: Optional[str]
        Method for removing outliers from the point cloud.
    height_attr: Optional[str]
        The COCO attribute name containing height information.
    """

    AVAILABLE_TASKS: Set[str] = {"classification", "height", "segmentation"}

    def __init__(
        self,
        dataset_name: str,
        fold: str,
        root_path: Union[str, List[str], Path, List[Path]],
        modalities: List[str] = ["image", "point_cloud", "dsm"],
        tasks: List[str] = ["classification"],
        voxel_size: Optional[float] = None,
        num_points_file: Optional[int] = None,
        downsample_method_file: Optional[str] = None,
        augment: Optional[Any] = None,
        num_points: Optional[int] = None,
        few_shot_path: Optional[str] = None,
        num_downsample_seeds: Optional[int] = 1,
        exclude_classes: Optional[List[int]] = None,
        outlier_removal: Optional[str] = None,
        height_attr: Optional[str] = None,
    ) -> None:
        """
        Initializes the multimodal dataset and validates task compatibility.

        Parameters
        ----------
        (See Class Docstring for parameter details)

        Returns
        -------
        None
        """
        unsupported = set(tasks) - self.AVAILABLE_TASKS
        if unsupported:
            raise ValueError(f"Unsupported tasks: {unsupported}. Allowed: {self.AVAILABLE_TASKS}")

        self.height_attr = height_attr
        other_attributes_names_to_pass = []
        if "height" in tasks and height_attr:
            other_attributes_names_to_pass.append(height_attr)

        super().__init__(
            fold=fold,
            root_path=root_path,
            transform=None,
            other_attributes_names_to_pass=other_attributes_names_to_pass,
            keep_unlabeled=False,
            few_shot_path=few_shot_path,
            exclude_classes=exclude_classes,
        )

        self.dataset_name = dataset_name
        self.modalities = modalities
        self.tasks = tasks
        self.augment = augment
        self.num_points = num_points
        self.num_downsample_seeds = num_downsample_seeds
        self.outlier_removal = outlier_removal

        self._initialize_modality_paths(voxel_size, num_points_file, downsample_method_file)

    def _initialize_modality_paths(
        self,
        voxel_size: Optional[float],
        num_points_file: Optional[int],
        downsample_method_file: Optional[str],
    ) -> None:
        """
        Populates tile metadata with associated file paths for different modalities.

        Parameters
        ----------
        voxel_size : float, optional
            Voxel size for point cloud processing.
        num_points_file : int, optional
            Point count target for the files.
        downsample_method_file : str, optional
            Method used for file-level downsampling.

        Returns
        -------
        None
        """
        for tile in self.tiles.values():
            base_path = Path(tile["path"])

            if "point_cloud" in self.modalities:
                raise NotImplementedError("Point cloud modality is not yet implemented in this dataset class.")

            if "dsm" in self.modalities:
                tile["dsm_path"] = self._derive_dsm_path(base_path)

    def _derive_dsm_path(self, raster_path: Path) -> Path:
        """
        Resolves the Digital Surface Model (DSM) path based on the dataset-specific naming convention.

        Parameters
        ----------
        raster_path : Path
            The file path to the original RGB raster tile.

        Returns
        -------
        Path
            The resolved path to the corresponding DSM file.

        Raises
        -------
        ValueError
            If the dataset name is unknown or the filename pattern cannot be parsed.
        """

        if self.dataset_name == "quebec_plantations":
            return Path(str(raster_path).replace("_rgb", "_dsm"))

        elif self.dataset_name == "quebec_trees":
            m = re.search(r"(\d{4})_(\d{2})_(\d{2})_([a-zA-Z]+)_(z\d+)_rgb", raster_path.name)
            if not m:
                raise ValueError(f"Could not parse quebec_trees pattern from {raster_path.name}")

            # Reconstruct prefix: YYYYMMDD_sitezn_p4rtk_dsm
            date_str, site_str = f"{m[1]}{m[2]}{m[3]}", f"{m[4]}{m[5]}"
            new_prefix = f"{date_str}_{site_str}_p4rtk_dsm"
            old_prefix = f"{m[1]}_{m[2]}_{m[3]}_{m[4]}_{m[5]}_rgb"
            return Path(str(raster_path).replace(old_prefix, new_prefix))

        elif self.dataset_name == "bci":
            return Path(str(raster_path).replace("_orthomosaic", "_dsm"))

        else:
            raise ValueError(f"Unknown dataset {self.dataset_name}")

    def _get_image_data(self, tile_info: Dict[str, Any]) -> Tuple[np.ndarray, Any, Tuple[float, float]]:
        """
        Reads, normalizes, and extracts geospatial metadata from an RGB raster tile.

        Parameters
        ----------
        tile_info : Dict[str, Any]
            Dictionary containing the 'path' key to the raster file.

        Returns
        -------
        img : np.ndarray
            The image data as a float32 array in (C, H, W) format, normalized to [0, 1].
        transform : rasterio.affine.Affine
            The affine transform mapping pixel coordinates to geographic coordinates.
        resolution : Tuple[float, float]
            The pixel resolution (x_res, y_res) of the tile.

        Raises
        -------
        ValueError
            If the image bit-depth is not 8-bit (uint8).
        """

        with rasterio.open(tile_info["path"]) as tile_file:
            # Handle band extraction
            if tile_file.count >= 3:
                img = tile_file.read([1, 2, 3])
            else:
                img = tile_file.read()
                if img.shape[0] == 1:
                    img = np.repeat(img, 3, axis=0)
            transform = tile_file.transform
            resolution = tile_file.res

        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        else:
            raise ValueError(f"Expected uint8, found {img.dtype} in {tile_info['path']}")

        return img, transform, resolution

    def _get_dsm_data(self, tile_info: Dict[str, Any]) -> Tuple[np.ndarray, float, float]:
        """
        Reads, normalizes, and scales the Digital Surface Model (DSM) raster.

        Parameters
        ----------
        tile_info : Dict[str, Any]
            Dictionary containing the 'dsm_path' key.

        Returns
        -------
        dsm_normalized : np.ndarray
            A float32 array normalized to the [0, 1] range.
        dsm_offset : float
            The minimum value of the original DSM (used for inversion/scaling).
        dsm_scale : float
            The range (max - min) of the original DSM.
        """
        with rasterio.open(tile_info["dsm_path"]) as dsm_file:
            dsm = dsm_file.read().astype(np.float32)

        dsm_offset = np.nanmin(dsm)
        dsm_max = np.nanmax(dsm)
        dsm_scale = dsm_max - dsm_offset

        if dsm_scale == 0:
            dsm_normalized = np.zeros_like(dsm)
        else:
            dsm_normalized = (dsm - dsm_offset) / dsm_scale

        return dsm_normalized, dsm_offset, dsm_scale

    def __getitem__(self, idx: int) -> Tuple[
        Dict[str, Any],
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Dict[str, Any],
    ]:
        """
        Retrieves a complete multimodal data sample including RGB image, point cloud, DSM, and associated targets for a
        specific index.

        Parameters
        ----------
        idx : int
            The index of the tile to retrieve from the dataset.

        Returns
        -------
        meta : Dict[str, Any]
            Metadata including filename, resolution, and normalization scales.
        img : np.ndarray
            The RGB image tile as a float32 array (C, H, W).
        points : np.ndarray, optional
            The sampled and normalized point cloud (N, 6) or None if not requested.
        dsm : np.ndarray, optional
            The normalized DSM tile (1, H, W) or None if not requested.
        targets : Dict[str, Any]
            Dictionary containing task-specific labels (class, height, segmentation).

        Raises
        -------
        IndexError
            If the index is out of bounds.
        ValueError
            If the tile contains no labels or unexpected data formats.
        NotImplementedError
            If a tile contains multiple labels (multi-label not supported).
        """
        tile_info = self.tiles[idx]

        img, _, resolution = self._get_image_data(tile_info)

        labels = tile_info.get("labels", [])
        if not labels:
            raise ValueError(f"No labels found for tile: {tile_info['path']}")
        if len(labels) > 1:
            raise NotImplementedError("Multi-label tasks are not yet supported.")

        label = labels[0]

        points, xy_scale = None, None
        if "point_cloud" in self.modalities:
            raise NotImplementedError("Point cloud modality is not yet implemented in this dataset class.")

        dsm, dsm_offset, dsm_scale = None, None, None
        if "dsm" in self.modalities:
            dsm, dsm_offset, dsm_scale = self._get_dsm_data(tile_info)

        mask = None
        if "segmentation" in self.tasks:
            mask_raw = decode_coco_segmentation(label, "mask")
            if not isinstance(mask_raw, np.ndarray) or mask_raw.ndim != 2:
                raise ValueError("Expected 2D array for segmentation mask")
            mask = mask_raw[np.newaxis, :, :]  # Expand to (1, H, W)

        if self.augment:
            img, points, dsm, mask = self.augment(img, points, dsm, mask)

        targets = {}
        if "classification" in self.tasks:
            targets["class"] = label["category_id"]

        if "height" in self.tasks:
            other_attrs = self._get_other_attributes_to_pass(idx)
            targets["height"] = other_attrs[self.height_attr][0]

        if "segmentation" in self.tasks:
            targets["segmentation"] = mask

        meta = {
            "filename": Path(tile_info["path"]).stem,
            "resolution": resolution,
            "xy_scale": float(xy_scale) if xy_scale is not None else None,
            "dsm_offset": float(dsm_offset) if dsm_offset is not None else None,
            "dsm_scale": float(dsm_scale) if dsm_scale is not None else None,
        }

        return meta, img, points, dsm, targets
