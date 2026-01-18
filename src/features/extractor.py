import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as ndi_label
from scipy.ndimage import convolve
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract morphological features from vessel segmentation"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def extract(self, binary_mask: np.ndarray) -> Dict[str, Any]:
        """
        Extract all vessel features from binary mask

        Args:
            binary_mask: Binary segmentation mask (0 or 1)

        Returns:
            Dictionary containing all extracted features
        """
        try:
            if binary_mask.size == 0:
                return self._get_empty_features()

            h, w = binary_mask.shape

            # 1. Basic vessel density
            vessel_density = float(np.mean(binary_mask))

            # 2. Clean the mask
            mask_clean = self._clean_mask(binary_mask)

            # 3. Skeletonization for morphological analysis
            from skimage.morphology import skeletonize
            skeleton = skeletonize(mask_clean > 0)

            # 4. Distance transform for width analysis
            distance_map = distance_transform_edt(mask_clean)

            # Get vessel widths at skeleton points
            skeleton_points = skeleton > 0
            widths = distance_map[skeleton_points] * 2  # Multiply by 2 for diameter

            # 5. Width-based features
            if len(widths) > 0:
                mean_width = float(np.mean(widths))
                std_width = float(np.std(widths))
                median_width = float(np.median(widths))
                width_iqr = float(np.percentile(widths, 75) - np.percentile(widths, 25))
                thin_ratio = float(np.sum(widths <= self.config.THIN_THRESHOLD) / len(widths))
                thick_ratio = float(np.sum(widths >= self.config.THICK_THRESHOLD) / len(widths))
            else:
                mean_width = std_width = median_width = width_iqr = 0.0
                thin_ratio = thick_ratio = 0.0

            # 6. Branching analysis
            branching_points = self._count_branching_points(skeleton)

            # 7. Tortuosity
            tortuosity = self._calculate_tortuosity(skeleton)

            # 8. Fractal dimension
            fractal_dimension = self._estimate_fractal_dimension(mask_clean)

            # 9. Regional analysis
            central_density, peripheral_density = self._calculate_regional_density(mask_clean)

            # 10. Vessel length
            vessel_length = float(np.sum(skeleton))

            # 11. Connected components
            labeled_mask, num_components = ndi_label(mask_clean)

            # 12. Component statistics
            component_areas = []
            for i in range(1, num_components + 1):
                component_areas.append(np.sum(labeled_mask == i))

            if component_areas:
                avg_component_area = float(np.mean(component_areas))
                max_component_area = float(np.max(component_areas))
            else:
                avg_component_area = max_component_area = 0.0

            # 13. Vessel directionality
            directionality = self._calculate_directionality(skeleton)

            # Compile all features
            features = {
                # Density metrics
                "vessel_density": vessel_density,
                "central_density": central_density,
                "peripheral_density": peripheral_density,

                # Width metrics
                "mean_width": mean_width,
                "std_width": std_width,
                "median_width": median_width,
                "width_iqr": width_iqr,
                "thin_ratio": thin_ratio,
                "thick_ratio": thick_ratio,

                # Morphological metrics
                "tortuosity": tortuosity,
                "fractal_dimension": fractal_dimension,
                "branching_points": int(branching_points),
                "vessel_length": vessel_length,
                "directionality": directionality,

                # Component metrics
                "num_components": int(num_components),
                "avg_component_area": avg_component_area,
                "max_component_area": max_component_area,

                # Raw data for visualization
                "width_distribution": widths.tolist() if len(widths) > 0 else []
            }

            self.logger.info(f"Extracted {len(features)} features")
            return features

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return self._get_empty_features()

    def _clean_mask(self, binary_mask: np.ndarray) -> np.ndarray:
        """Clean binary mask by removing small objects"""
        from skimage.morphology import remove_small_objects
        return remove_small_objects(
            binary_mask.astype(bool),
            min_size=self.config.MIN_VESSEL_AREA
        ).astype(np.uint8)

    def _count_branching_points(self, skeleton: np.ndarray) -> int:
        """Count branching points (endpoints + junctions)"""
        # Use convolution to count neighbors
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=np.uint8)

        neighbor_count = convolve(skeleton.astype(np.uint8), kernel, mode='constant')

        # Count endpoints (1 neighbor) and junctions (>=3 neighbors)
        skeleton_indices = skeleton > 0
        neighbor_values = neighbor_count[skeleton_indices]

        num_endpoints = np.sum(neighbor_values == 1)
        num_junctions = np.sum(neighbor_values >= 3)

        return int(num_endpoints + num_junctions)

    def _calculate_tortuosity(self, skeleton: np.ndarray) -> float:
        """Calculate vessel tortuosity"""
        if np.sum(skeleton) == 0:
            return 1.0

        # Label connected components
        labeled_skeleton, num_components = ndi_label(skeleton)

        tortuosities = []
        for i in range(1, num_components + 1):
            # Get coordinates of this component
            coords = np.argwhere(labeled_skeleton == i)

            if len(coords) < 2:
                continue

            # Curved length (number of pixels)
            curved_length = len(coords)

            # Straight line distance between endpoints
            start = coords[0]
            end = coords[-1]
            straight_length = np.linalg.norm(end - start)

            if straight_length > 0:
                tortuosity = curved_length / straight_length
                tortuosities.append(tortuosity)

        return float(np.mean(tortuosities)) if tortuosities else 1.0

    def _estimate_fractal_dimension(self, binary_mask: np.ndarray) -> float:
        """Estimate fractal dimension using box-counting method"""
        h, w = binary_mask.shape

        # Box sizes (powers of 2)
        box_sizes = [2, 4, 8, 16, 32, 64]
        box_sizes = [s for s in box_sizes if s <= min(h, w)]

        if len(box_sizes) < 2:
            return 1.0

        counts = []

        for size in box_sizes:
            num_boxes = 0

            # Count boxes containing vessel pixels
            for y in range(0, h, size):
                for x in range(0, w, size):
                    box = binary_mask[y:min(y + size, h), x:min(x + size, w)]
                    if np.sum(box) > 0:
                        num_boxes += 1

            counts.append(num_boxes)

        counts = np.array(counts)
        box_sizes = np.array(box_sizes)

        # Filter out zero counts
        valid = counts > 0

        if np.sum(valid) < 2:
            return 1.0

        try:
            # Linear regression on log-log plot
            coefficients = np.polyfit(
                np.log(1.0 / box_sizes[valid]),
                np.log(counts[valid]),
                1
            )
            fractal_dim = -coefficients[0]

            # Ensure reasonable range
            return max(1.0, min(fractal_dim, 2.5))

        except:
            return 1.0

    def _calculate_regional_density(self, binary_mask: np.ndarray) -> Tuple[float, float]:
        """Calculate central and peripheral vessel density"""
        h, w = binary_mask.shape

        # Central region (middle 50%)
        y_start, y_end = h // 4, 3 * h // 4
        x_start, x_end = w // 4, 3 * w // 4

        central_region = binary_mask[y_start:y_end, x_start:x_end]
        central_area = central_region.size
        central_density = float(np.sum(central_region) / central_area) if central_area > 0 else 0.0

        # Peripheral region (everything outside central)
        total_area = h * w
        peripheral_area = total_area - central_area

        if peripheral_area > 0:
            total_vessels = np.sum(binary_mask)
            central_vessels = np.sum(central_region)
            peripheral_vessels = total_vessels - central_vessels
            peripheral_density = float(peripheral_vessels / peripheral_area)
        else:
            peripheral_density = 0.0

        return central_density, peripheral_density

    def _calculate_directionality(self, skeleton: np.ndarray) -> float:
        """Calculate vessel directionality (anisotropy)"""
        if np.sum(skeleton) == 0:
            return 0.0

        # Calculate gradient directions
        from scipy.ndimage import sobel

        # Compute gradients
        gy = sobel(skeleton.astype(float), axis=0)
        gx = sobel(skeleton.astype(float), axis=1)

        # Calculate angles
        angles = np.arctan2(gy, gx)
        angles = angles[skeleton > 0]

        if len(angles) == 0:
            return 0.0

        # Calculate circular variance (1 - |mean vector|)
        mean_cos = np.mean(np.cos(angles))
        mean_sin = np.mean(np.sin(angles))
        mean_vector_length = np.sqrt(mean_cos ** 2 + mean_sin ** 2)

        return float(mean_vector_length)

    def _get_empty_features(self) -> Dict[str, Any]:
        """Return empty feature dictionary"""
        return {
            "vessel_density": 0.0,
            "central_density": 0.0,
            "peripheral_density": 0.0,
            "mean_width": 0.0,
            "std_width": 0.0,
            "median_width": 0.0,
            "width_iqr": 0.0,
            "thin_ratio": 0.0,
            "thick_ratio": 0.0,
            "tortuosity": 1.0,
            "fractal_dimension": 1.0,
            "branching_points": 0,
            "vessel_length": 0.0,
            "directionality": 0.0,
            "num_components": 0,
            "avg_component_area": 0.0,
            "max_component_area": 0.0,
            "width_distribution": []
        }