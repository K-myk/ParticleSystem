"""
config.py
高精度配置文件 - 仅SAM模式
"""

import os
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

@dataclass
class PreprocessConfig:
    """预处理配置"""
    resize_enabled: bool = True
    max_dimension: int = 1024

    convert_grayscale: bool = True
    denoise_enabled: bool = True
    denoise_method: str = "median"
    gaussian_kernel: Tuple[int, int] = (5, 5)
    median_kernel: int = 5
    enhance_contrast: bool = True
    clahe_clip_limit: float = 4.0
    clahe_grid_size: Tuple[int, int] = (4, 4)

@dataclass
class DetectionConfig:
    """检测配置 - 仅SAM模式"""

    # --- SAM 模型配置 ---
    sam_model_type: str = "vit_b"
    sam_checkpoint: str = "weights/sam_vit_b_01ec64.pth"
    sam_device: Optional[str] = None

    sam_points_per_side: int = 32
    sam_pred_iou_thresh: float = 0.80
    sam_stability_score_thresh: float = 0.88

    sam_crop_n_layers: int = 0
    sam_min_mask_region_area: int = 50
    sam_overlap_threshold: float = 0.5
    sam_use_morphology: bool = True

    # --- 通用筛选 ---
    min_area: float = 50
    max_area: float = 10000000
    min_circularity: float = 0.1
    max_circularity: float = 1.0

@dataclass
class MeasurementConfig:
    pixel_to_mm: float = 0.1

@dataclass
class VisualizationConfig:
    contour_color: Tuple[int, int, int] = (0, 255, 0)
    text_color: Tuple[int, int, int] = (0, 0, 255)
    centroid_color: Tuple[int, int, int] = (0, 0, 255)
    contour_thickness: int = 2
    font_scale: float = 0.5
    show_id: bool = True
    show_area: bool = False
    show_centroid: bool = True

@dataclass
class SystemConfig:
    input_dir: str = "data/images"
    output_dir: str = "results"
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    def __post_init__(self):
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)