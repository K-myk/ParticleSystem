# modules/statistical_analyzer.py
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any

@dataclass
class StatisticalResult:
    count: int = 0
    area_mean: float = 0.0
    diameter_mean: float = 0.0
    diameter_std: float = 0.0
    diameter_d10: float = 0.0
    diameter_d50: float = 0.0
    diameter_d90: float = 0.0
    size_distribution: Dict[str, int] = field(default_factory=dict)

class StatisticalAnalyzer:
    def analyze(self, measurements: List) -> StatisticalResult:
        if not measurements:
            return StatisticalResult()

        areas = [m.area_mm2 for m in measurements]
        diameters = [m.equivalent_diameter_mm for m in measurements]
        
        count = len(measurements)
        d_arr = np.array(diameters)
        
        res = StatisticalResult()
        res.count = count
        res.area_mean = np.mean(areas) if count > 0 else 0
        
        if count > 0:
            res.diameter_mean = np.mean(d_arr)
            res.diameter_std = np.std(d_arr)
            res.diameter_d10 = np.percentile(d_arr, 10)
            res.diameter_d50 = np.percentile(d_arr, 50)
            res.diameter_d90 = np.percentile(d_arr, 90)
            
            # 简易粒径分布
            bins = [0, 2, 5, 10, 20, 50, 9999]
            labels = ["<2mm", "2-5mm", "5-10mm", "10-20mm", "20-50mm", ">50mm"]
            hist, _ = np.histogram(d_arr, bins)
            res.size_distribution = dict(zip(labels, hist.tolist()))
            
        return res

    def generate_report(self, res: StatisticalResult) -> str:
        lines = [
            "=== 统计分析报告 ===",
            f"颗粒总数: {res.count}",
            f"平均面积: {res.area_mean:.4f} mm²",
            f"平均直径: {res.diameter_mean:.4f} mm (Std: {res.diameter_std:.4f})",
            f"D10: {res.diameter_d10:.4f} mm",
            f"D50 (中位径): {res.diameter_d50:.4f} mm",
            f"D90: {res.diameter_d90:.4f} mm",
            "\n[粒径分布]"
        ]
        for k, v in res.size_distribution.items():
            if v > 0:
                lines.append(f"  {k}: {v} 个")
        return "\n".join(lines)
    
    def export_to_dict(self, res: StatisticalResult) -> dict:
        return {
            "Total Count": res.count,
            "Mean Diameter (mm)": res.diameter_mean,
            "D50 (mm)": res.diameter_d50
        }