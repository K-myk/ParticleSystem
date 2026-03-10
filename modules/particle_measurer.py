"""
modules/particle_measurer.py
颗粒物测量模块
修复：长宽比计算逻辑，防止出现全是 0 的情况
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Any
import math


@dataclass
class MeasurementData:
    """测量结果数据结构"""
    particle_id: int
    # 像素单位
    area_pixels: float
    perimeter_pixels: float
    # 物理单位 (mm)
    area_mm2: float
    perimeter_mm: float
    equivalent_diameter_mm: float
    # 形状因子
    circularity: float
    aspect_ratio: float
    solidity: float
    # 几何参数
    major_axis_mm: float = 0.0
    minor_axis_mm: float = 0.0
    feret_max_mm: float = 0.0
    centroid_x: float = 0.0
    centroid_y: float = 0.0


@dataclass
class BatchMeasurementResult:
    measurements: List[MeasurementData] = field(default_factory=list)
    pixel_to_mm: float = 0.1


class ParticleMeasurer:
    def __init__(self, config):
        self.config = config
        self.pixel_to_mm = config.pixel_to_mm

    def measure_batch(self, particles: List, image_shape: tuple) -> BatchMeasurementResult:
        results = []
        scale = self.pixel_to_mm

        for p in particles:
            # 1. 计算物理尺寸
            area_mm2 = p.area * (scale ** 2)
            perimeter_mm = p.perimeter * scale
            equiv_d_mm = p.equivalent_diameter * scale

            # 2. 拟合椭圆长短轴 (修复长宽比来源)
            major_axis = 0.0
            minor_axis = 0.0
            aspect_ratio = 0.0

            if p.fitted_ellipse:
                (cx, cy), (ax1, ax2), angle = p.fitted_ellipse
                major_axis = max(ax1, ax2)
                minor_axis = min(ax1, ax2)
                # 计算长宽比 (短轴/长轴，范围 0~1)
                if major_axis > 0:
                    aspect_ratio = minor_axis / major_axis
            else:
                # 如果无法拟合椭圆，退化为外接矩形的长宽比
                x, y, w, h = p.bounding_rect
                if h > 0:
                    # 统一规范：长宽比取 (短边/长边)，确保在 0-1 之间
                    aspect_ratio = min(w, h) / max(w, h)

            # 3. Feret 直径 (最大卡尺径)
            feret_max = 0
            if p.convex_hull is not None and len(p.convex_hull) > 2:
                rect = cv2.minAreaRect(p.convex_hull)
                feret_max = max(rect[1])

            data = MeasurementData(
                particle_id=p.id,
                area_pixels=p.area,
                perimeter_pixels=p.perimeter,
                area_mm2=area_mm2,
                perimeter_mm=perimeter_mm,
                equivalent_diameter_mm=equiv_d_mm,
                circularity=p.circularity,
                aspect_ratio=aspect_ratio,  # 修复后的值
                solidity=p.solidity,
                major_axis_mm=major_axis * scale,
                minor_axis_mm=minor_axis * scale,
                feret_max_mm=feret_max * scale,
                centroid_x=p.centroid[0],
                centroid_y=p.centroid[1]
            )
            results.append(data)

        return BatchMeasurementResult(measurements=results, pixel_to_mm=scale)

    def export_measurements(self, result: BatchMeasurementResult) -> List[dict]:
        """导出为字典列表，用于保存 Excel"""
        return [{
            "ID": m.particle_id,
            "面积(mm²)": round(m.area_mm2, 4),
            "周长(mm)": round(m.perimeter_mm, 4),
            "等效直径(mm)": round(m.equivalent_diameter_mm, 4),
            "圆度": round(m.circularity, 4),
            "长宽比": round(m.aspect_ratio, 4),
            "最大Feret径(mm)": round(m.feret_max_mm, 4)
        } for m in result.measurements]