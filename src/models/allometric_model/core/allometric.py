from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import torch
from affine import Affine
from shapely import Polygon
from shapely.geometry import shape
from torch import nn


class Allometric(nn.Module):
    """
    Allometric model to predict tree height from crown segmentation.

    This module implements height estimation based on the relationship:
    Height = exp(slope * log(crown_radius) + intercept)
    """

    def __init__(
        self,
        radius_determination: str,
        allometry_path: Path,
        dataset_name: str,
        category_dict: dict[int, str],
    ) -> None:
        """
        Initialize the Allometric model.

        Args:
            radius_determination: Strategy to calculate radius ('area_circle' or 'rotated_rectangle').
            allometry_path: Path to the CSV file containing allometry constants.
            dataset_name: Name of the dataset for species remapping.
            category_dict: Mapping of class IDs to species names.
        """
        super().__init__()
        self.radius_determination = radius_determination
        self.allometry_params_registry = AllometryParamsRegistry(
            allometry_path=allometry_path,
            dataset_name=dataset_name,
            relationship_type="log_height_vs_log_crown_radius",
            category_dict=category_dict,
        )

    def forward(
        self, category_id: torch.Tensor, segmentation: torch.Tensor, meta: list[dict[str, Any]]
    ) -> torch.Tensor:
        """
        Predict height based on segmentation masks and species IDs.

        Args:
            category_id: Tensor of shape (B,) containing species class IDs.
            segmentation: Tensor of shape (B, 1, H, W) or (B, H, W) containing masks.
            meta: List of dictionaries containing 'resolution'.

        Returns:
            Tensor of shape (B,) containing predicted heights in meters.
        """
        if self.radius_determination == "area_circle":
            crown_radius = self.compute_radius_from_mask_circle(segmentation, meta)
        elif self.radius_determination == "rotated_rectangle":
            crown_radius = self.compute_radius_from_mask_rectangle(segmentation, meta)
        else:
            raise ValueError(f"Unknown determination method: {self.radius_determination}")
        slope, intercept = self.allometry_params_registry.get_params(category_id)
        log_height = slope * torch.log(crown_radius) + intercept
        return torch.exp(log_height)

    def compute_radius_from_mask_rectangle(self, mask_tensor: torch.Tensor, meta: list[dict[str, Any]]) -> torch.Tensor:
        """
        Estimates crown radius by finding the minimum rotated bounding box.

        Args:
            mask_tensor: The mask batch to be processed.
            meta: The metadata list used to calculate the physical area of
                individual pixels.

        Returns:
            The calculated radius for each mask in the batch.
        """
        device = mask_tensor.device
        if mask_tensor.ndim == 4:
            mask_tensor = mask_tensor.squeeze(1)

        masks_np = mask_tensor.detach().cpu().numpy().astype(np.uint8)
        batch_radii: list[float] = []

        for i, mask in enumerate(masks_np):
            res = meta[i]["resolution"]
            x_res, y_res = res
            transform = Affine(x_res, 0.0, 0.0, 0.0, y_res, 0.0)
            if mask.max() == 0:
                raise ValueError(f"Batch index {i} contains 0 shapes (mask is empty).")

            shapes_gen = rasterio.features.shapes(mask, mask=mask > 0, transform=transform)
            polygons = [shape(geom) for geom, _ in shapes_gen]

            if not polygons:
                raise ValueError(f"Batch index {i} contains 0 shapes after polygonization.")

            main_poly = max(polygons, key=lambda p: p.area)
            rect = main_poly.minimum_rotated_rectangle

            if not isinstance(rect, Polygon):
                raise ValueError(
                    f"Batch index {i} resulted in degenerate geometry: '{type(rect).__name__}'. "
                    "A valid Polygon with non-zero area is required."
                )

            coords = list(rect.exterior.coords)
            side_a = np.linalg.norm(np.array(coords[1]) - np.array(coords[0])).item()
            side_b = np.linalg.norm(np.array(coords[2]) - np.array(coords[1])).item()

            batch_radii.append((side_a + side_b) / 4.0)

        return torch.tensor(batch_radii, dtype=torch.float32, device=device)

    def compute_radius_from_mask_circle(self, mask_tensor: torch.Tensor, meta: list[dict[str, Any]]) -> torch.Tensor:
        """
        Estimates crown radius by calculating the area of the mask and inverting the circle area formula.

        Args:
            mask_tensor: The mask batch where the sum of pixels represents the area.
            meta: The metadata list used to calculate the physical area of
                individual pixels.

        Returns:
            The calculated radius for each mask in the batch.
        """
        device = mask_tensor.device
        if mask_tensor.ndim == 4:
            mask_tensor = mask_tensor.squeeze(1)

        res_areas = [m["resolution"][0] * m["resolution"][1] for m in meta]
        pixel_area_vec = torch.tensor(res_areas, dtype=torch.float32, device=device)

        total_pixels = mask_tensor.sum(dim=(1, 2))
        if (total_pixels == 0).any():
            bad_indices = (total_pixels == 0).nonzero(as_tuple=True)[0].tolist()
            print(f"Batch indices {bad_indices} are empty (0 shapes). Setting number of pixels to 1.")
            total_pixels = total_pixels.clamp(min=1)
        total_area_m2 = total_pixels * pixel_area_vec

        radii = torch.sqrt(total_area_m2 / torch.pi)

        return radii


class AllometryParamsRegistry:
    """
    A registry to manage and retrieve allometric parameters.

    It pre-loads the CSV and pre-computes the mapping from 'Category ID' to '(Slope, Intercept)' during initialization
    to maximize batch processing speed.
    """

    DATASET_MAPPINGS: Final[dict[str, dict[str, str]]] = {
        "quebec_trees": {
            "Thuja_occidentalis_L": "Thuja occidentalis",
            "Abies_balsamea_(L)_Mill": "Abies balsamea",
            "Larix_laricina_(Du_Roi)_KKoch": "Larix laricina",
            "Tsuga_canadensis_(L)": "Tsuga canadensis",
            "Fagus_grandifolia_Ehrh": "Fagus grandifolia",
            "Populus_L": "Populus",
            "Acer_pensylvanicum_L": "Acer pensylvanicum",
            "Acer_saccharum_Marshall": "Acer saccharum",
            "Acer_rubrum_L": "Acer rubrum",
            "Pinus_strobus_L": "Pinus strobus",
            "Betula_alleghaniensis_Britton": "Betula alleghaniensis",
            "Betula_papyrifera_Marshall": "Betula papyrifera",
            "Picea_ADietr": "Picea",
            "dead": "all",
        },
        "quebec_plantations": {
            "piba": "Pinus banksiana",
            "pima": "Picea mariana",
            "pist": "Pinus strobus",
            "pigl": "Picea glauca",
            "thoc": "Thuja occidentalis",
            "ulam": "Ulmus americana",
            "beal": "Betula alleghaniensis",
            "acsa": "Acer saccharum",
        },
    }

    def __init__(
        self, allometry_path: Path, dataset_name: str, relationship_type: str, category_dict: dict[int, str]
    ) -> None:
        """
        Initialize the registry and pre-compute parameter caches.

        Args:
            allometry_path: Path to the CSV containing 'category', 'relationship',
                'slope', and 'intercept'.
            dataset_name: Key for the dataset mapping (e.g., 'quebec_trees').
            relationship_type: The relationship filter (e.g., 'height_to_dbh').
            category_dict: Map of integer IDs to string names from the dataset.
        """
        self.dataset_name = dataset_name
        self.relationship_type = relationship_type

        if dataset_name == "bci":
            self.current_remap = None
        elif dataset_name not in self.DATASET_MAPPINGS:
            raise ValueError(f"Unknown dataset_name: '{dataset_name}'.")
        else:
            self.current_remap = self.DATASET_MAPPINGS[dataset_name]

        self._load_and_process_csv(allometry_path)
        self.id_to_params = self._build_lookup_cache(category_dict)

    def _load_and_process_csv(self, path: Path) -> None:
        """
        Internal method to load CSV and validate data.

        Args:
            path: Filesystem path to the allometry CSV.
        """
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find allometry file at: {path}")

        required_cols = ["category", "relationship", "slope", "intercept"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")

        df = df[df["relationship"] == self.relationship_type]

        if df.empty:
            raise ValueError(f"No entries found in CSV with relationship='{self.relationship_type}'.")

        if df.duplicated(subset=["category"]).any():
            duplicated_cats = df[df.duplicated(subset=["category"])]["category"].unique().tolist()
            raise ValueError(
                f"Duplicate categories found in CSV for relationship '{self.relationship_type}': {duplicated_cats}.\n"
                "Each category must appear exactly once per relationship type."
            )

        self.csv_lookup = df.set_index("category")[["slope", "intercept"]].to_dict("index")

    def _build_lookup_cache(self, category_dict: dict[int, str]) -> dict[int, tuple[float, float]]:
        """
        Internal method to map Input IDs -> CSV Params.

        Args:
            category_dict: Mapping of dataset IDs to string species names.

        Returns:
            A dictionary mapping IDs to (slope, intercept) tuples.
        """
        cache = {}

        for cat_id, cat_name in category_dict.items():
            cat_id = int(cat_id)

            if self.dataset_name == "bci":
                csv_target_name = cat_name
                if cat_name == "Arecaceae" or cat_name == "Cordiaceae":
                    csv_target_name = "all"
            else:
                if self.current_remap is None or cat_name not in self.current_remap:
                    raise ValueError(
                        f"The category '{cat_name}' (ID: {cat_id}) is missing from the "
                        f"mapping configuration for dataset '{self.dataset_name}'."
                    )

                csv_target_name = self.current_remap[cat_name]

            if csv_target_name not in self.csv_lookup:
                available_keys = list(self.csv_lookup.keys())[:5]
                raise ValueError(
                    f"Mapped category '{csv_target_name}' (from '{cat_name}') not found "
                    f"for relationship '{self.relationship_type}'.\n"
                    f"Available categories start with: {available_keys}..."
                )

            params = self.csv_lookup[csv_target_name]
            cache[cat_id] = (params["slope"], params["intercept"])

        return cache

    def get_params(self, category_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the parameters for a batch of IDs using the pre-computed cache.

        Args:
            category_ids (torch.Tensor): Tensor of shape (B,) with category IDs.

        Returns:
            tuple: (slopes, intercepts) as torch.Tensors.
        """
        slopes = []
        intercepts = []

        device = category_ids.device
        ids_np = category_ids.detach().cpu().numpy()

        for cat_id in ids_np:
            cat_id = int(cat_id)

            if cat_id not in self.id_to_params:
                raise ValueError(
                    f"Category ID {cat_id} found in input batch but was not present in the initialization category_dict."
                )

            s, i = self.id_to_params[cat_id]
            slopes.append(s)
            intercepts.append(i)

        return (
            torch.tensor(slopes, dtype=torch.float32, device=device),
            torch.tensor(intercepts, dtype=torch.float32, device=device),
        )
