from typing import Any

import albumentations as A
import numpy as np


class ImageAugmentor:
    """Albumentations-based augmentor for synchronized geometric
    transformations.

    Applies D4 group transformations (rotations/reflections) to multi-
    modal geospatial data including images, DSMs, and masks.
    """

    def __init__(
        self,
        geo_p: float = 1.0,
    ) -> None:
        """Initialize the augmentor with a replayable transformation pipeline.

        Args:
            geo_p: Probability of applying the geometric transformations.
        """
        self.geo_transform = A.ReplayCompose([A.SquareSymmetry(p=geo_p)])

    def _to_hwc(self, data: np.ndarray) -> np.ndarray:
        """Convert [C, H, W] to [H, W, C] for Albumentations compatibility."""
        return np.transpose(data, (1, 2, 0))

    def _to_chw(self, data: np.ndarray) -> np.ndarray:
        """Convert [H, W, C] back to [C, H, W] for downstream processing."""
        return np.transpose(data, (2, 0, 1))

    def _apply_replay(
        self, arr: np.ndarray | None, replay: dict[str, Any]
    ) -> np.ndarray | None:
        """Apply a recorded transformation to a single array using replay
        logic.

        Args:
            arr: Input array or None.
            replay: The 'replay' metadata from the initial transformation.

        Returns:
            The transformed array in CHW format, or None.
        """
        if arr is None:
            return None

        # Convert to HWC, apply replay, and convert back to CHW
        arr_hwc = self._to_hwc(arr)
        augmented = self.geo_transform.replay(replay, image=arr_hwc)

        return self._to_chw(augmented["image"])

    def __call__(
        self,
        img: np.ndarray,
        points: np.ndarray | None,
        dsm: np.ndarray | None,
        mask: np.ndarray | None,
    ) -> tuple[np.ndarray, None, np.ndarray | None, np.ndarray | None]:
        """Synchronize augmentations across image, DSM, and mask.

        Args:
            img: Image array [C, H, W].
            points: Point cloud array [N, 3 or 6].
            dsm: DSM array [1, H, W].
            mask: Mask array [1, H, W].

        Returns:
            Tuple containing (img_aug, None, dsm_aug, mask_aug).

        Raises:
            NotImplementedError: If points is not None, as coordinate
                transformation is not yet implemented.
        """
        if points is not None:
            raise NotImplementedError(
                "Point cloud augmentation is not yet supported. "
                "Geometric transforms on [N, 3] coordinates require "
                "manual rotation matrix application."
            )

        # 1. Primary transformation on image to generate the replay state
        img_hwc = self._to_hwc(img)
        augmented = self.geo_transform(image=img_hwc)

        img_aug = self._to_chw(augmented["image"])
        replay = augmented["replay"]

        # 2. Replay transformations on other spatial modalities
        dsm_aug = self._apply_replay(dsm, replay)
        mask_aug = self._apply_replay(mask, replay)

        return img_aug, None, dsm_aug, mask_aug
