"""
modules/particle_detector.py
颗粒物检测模块 - 仅SAM模式
"""

import cv2
import numpy as np
import time
import logging
import torch
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Particle:
    id: int
    contour: np.ndarray
    area: float
    perimeter: float
    centroid: Tuple[float, float]
    bounding_rect: Tuple[int, int, int, int]
    equivalent_diameter: float = 0.0
    circularity: float = 0.0
    aspect_ratio: float = 0.0
    solidity: float = 0.0
    convex_hull: Optional[np.ndarray] = None
    min_enclosing_circle: Tuple[Tuple[float, float], float] = ((0.0, 0.0), 0.0)
    fitted_ellipse: Optional[Tuple] = None
    mask: Optional[np.ndarray] = None
    sam_score: float = 0.0


@dataclass
class DetectionResult:
    binary_image: np.ndarray
    morphed_image: np.ndarray
    particles: List[Particle]
    contours: List[np.ndarray]
    filtered_count: int
    processing_time: float = 0.0


class ParticleDetector:
    """SAM 颗粒物检测器"""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.mask_generator = None
        self.device = None
        self._init_sam_model()

    def _init_sam_model(self):
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        except ImportError:
            logger.error("未安装 segment_anything 库，请执行: pip install git+https://github.com/facebookresearch/segment-anything.git")
            return

        # 设备检测
        if self.config.sam_device:
            self.device = self.config.sam_device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"正在加载 SAM 模型 ({self.device})...")
        try:
            sam = sam_model_registry[self.config.sam_model_type](checkpoint=self.config.sam_checkpoint)
            sam.to(device=self.device)

            self.mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=self.config.sam_points_per_side,
                pred_iou_thresh=self.config.sam_pred_iou_thresh,
                stability_score_thresh=self.config.sam_stability_score_thresh,
                crop_n_layers=self.config.sam_crop_n_layers,
                min_mask_region_area=self.config.sam_min_mask_region_area,
            )
            logger.info("SAM 模型加载成功")
        except Exception as e:
            logger.error(f"SAM 模型加载失败: {e}")

    def detect(self, image: np.ndarray) -> DetectionResult:
        t_start = time.time()
        if self.mask_generator is None:
            logger.error("SAM 模型未初始化，无法进行检测")
            return self._empty_result(image.shape[:2])

        # BGR 转 RGB
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            masks_data = self.mask_generator.generate(image_rgb)
        except Exception as e:
            logger.error(f"SAM 推理错误: {e}")
            return self._empty_result(image.shape[:2])

        # 按预测置信度(IoU)降序排序
        if len(masks_data) > 0 and 'predicted_iou' in masks_data[0]:
            masks_data = sorted(masks_data, key=lambda x: x['predicted_iou'], reverse=True)
        else:
            masks_data = sorted(masks_data, key=lambda x: x['area'], reverse=True)

        h, w = image.shape[:2]
        occupied = np.zeros((h, w), dtype=bool)
        combined_binary = np.zeros((h, w), dtype=np.uint8)

        particles = []
        filtered_count = 0
        particle_id = 0

        for mask_info in masks_data:
            seg_mask = mask_info['segmentation']
            area = mask_info['area']

            # 基础面积过滤
            if area < self.config.min_area or area > self.config.max_area:
                filtered_count += 1
                continue

            # 重叠检测
            overlap = np.logical_and(seg_mask, occupied)
            overlap_ratio = np.sum(overlap) / area

            if overlap_ratio > self.config.sam_overlap_threshold:
                filtered_count += 1
                continue

            # 去除已占用区域
            if overlap_ratio > 0:
                seg_mask = np.logical_and(seg_mask, ~occupied)

            # 生成二值图
            mask_uint8 = (seg_mask.astype(np.uint8) * 255).copy()

            # 形态学处理：断开细微粘连
            if self.config.sam_use_morphology:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=1)

            # 轮廓提取
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                p = self._create_particle(particle_id, contour)
                if p:
                    particles.append(p)
                    particle_id += 1

                    cv2.drawContours(mask_uint8, [contour], -1, 255, -1)
                    mask_bool = mask_uint8 > 0
                    occupied = np.logical_or(occupied, mask_bool)
                    combined_binary = cv2.bitwise_or(combined_binary, mask_uint8)
                else:
                    filtered_count += 1

        return DetectionResult(
            binary_image=combined_binary,
            morphed_image=combined_binary.copy(),
            particles=particles,
            contours=[p.contour for p in particles],
            filtered_count=filtered_count,
            processing_time=time.time() - t_start
        )

    def _create_particle(self, pid, contour):
        area = cv2.contourArea(contour)
        if area < self.config.min_area: return None
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0: return None

        circularity = 4 * np.pi * area / (perimeter ** 2)
        if self.config.min_circularity > 0 and circularity < self.config.min_circularity:
            return None

        M = cv2.moments(contour)
        cx = M['m10'] / M['m00'] if M['m00'] > 0 else 0
        cy = M['m01'] / M['m00'] if M['m00'] > 0 else 0

        rect = cv2.boundingRect(contour)
        hull = cv2.convexHull(contour)
        min_circle = cv2.minEnclosingCircle(contour)

        try:
            ellipse = cv2.fitEllipse(contour) if len(contour) >= 5 else None
        except:
            ellipse = None

        return Particle(
            id=pid, contour=contour, area=area, perimeter=perimeter,
            centroid=(cx, cy), bounding_rect=rect,
            equivalent_diameter=np.sqrt(4 * area / np.pi),
            circularity=circularity,
            convex_hull=hull, min_enclosing_circle=min_circle,
            fitted_ellipse=ellipse
        )

    def _empty_result(self, shape):
        h, w = shape
        blank = np.zeros((h, w), dtype=np.uint8)
        return DetectionResult(blank, blank, [], [], 0, 0.0)

    def update_config(self, config):
        self.config = config

    def get_device_info(self):
        if self.device:
            return f"SAM ({self.device})"
        return "SAM (未初始化)"