# modules/image_preprocessor.py
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class PreprocessResult:
    original: np.ndarray
    grayscale: np.ndarray
    denoised: np.ndarray
    enhanced: np.ndarray
    final: np.ndarray
    steps: Dict[str, np.ndarray]

class ImagePreprocessor:
    def __init__(self, config):
        self.config = config

    def process(self, image: np.ndarray) -> PreprocessResult:
        steps = {"original": image.copy()}
        current = image.copy()

        # 1. 缩放
        if self.config.resize_enabled:
            h, w = current.shape[:2]
            max_dim = max(h, w)
            if max_dim > self.config.max_dimension:
                scale = self.config.max_dimension / max_dim
                current = cv2.resize(current, None, fx=scale, fy=scale)
            steps["resized"] = current.copy()

        # 2. 灰度化
        if self.config.convert_grayscale:
            if len(current.shape) == 3:
                grayscale = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            else:
                grayscale = current.copy()
        else:
            grayscale = current.copy()
        steps["grayscale"] = grayscale.copy()

        # 3. 降噪
        if self.config.denoise_enabled:
            if self.config.denoise_method == "median":
                denoised = cv2.medianBlur(grayscale, self.config.median_kernel)
            elif self.config.denoise_method == "gaussian":
                denoised = cv2.GaussianBlur(grayscale, self.config.gaussian_kernel, 0)
            else:
                denoised = grayscale
        else:
            denoised = grayscale
        steps["denoised"] = denoised.copy()

        # 4. 增强 (CLAHE)
        if self.config.enhance_contrast:
            clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=self.config.clahe_grid_size
            )
            enhanced = clahe.apply(denoised)
        else:
            enhanced = denoised
        steps["enhanced"] = enhanced.copy()

        return PreprocessResult(
            original=image,
            grayscale=grayscale,
            denoised=denoised,
            enhanced=enhanced,
            final=enhanced,
            steps=steps
        )