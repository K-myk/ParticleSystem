"""
modules/visualizer.py
可视化模块
新增：在每张统计图表下方添加详细的说明文字
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from typing import List, Any
import matplotlib

logger = logging.getLogger(__name__)

# 配置 Matplotlib 支持中文显示
font_names = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'Arial Unicode MS']
font_found = False
for font in font_names:
    try:
        matplotlib.rc('font', family=font)
        plt.figure()
        plt.close()
        font_found = True
        break
    except Exception:
        continue

plt.rcParams['axes.unicode_minus'] = False

class Visualizer:
    def __init__(self, config):
        self.config = config

    def draw_detection_result(self, image: np.ndarray, particles: list, measurements: list = None) -> np.ndarray:
        if len(image.shape) == 2:
            canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            canvas = image.copy()

        for i, p in enumerate(particles):
            cv2.drawContours(canvas, [p.contour], -1, self.config.contour_color, self.config.contour_thickness)
            if self.config.show_centroid:
                cx, cy = int(p.centroid[0]), int(p.centroid[1])
                cv2.circle(canvas, (cx, cy), 3, self.config.centroid_color, -1)

            label_parts = []
            if self.config.show_id: label_parts.append(f"#{p.id}")
            if measurements and i < len(measurements) and self.config.show_area:
                m = measurements[i]
                label_parts.append(f"A:{m.area_mm2:.1f}")

            if label_parts:
                label = " ".join(label_parts)
                cx, cy = int(p.centroid[0]), int(p.centroid[1])
                # 给文字加个黑边，防止在亮色石子上看不清
                cv2.putText(canvas, label, (cx - 10, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, (0,0,0), 3)
                cv2.putText(canvas, label, (cx - 10, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, self.config.text_color, 1)

        return canvas

    def generate_plots(self, measurements: List[Any]) -> List[np.ndarray]:
        if not measurements: return []

        plots = []
        diameters = [m.equivalent_diameter_mm for m in measurements]
        areas = [m.area_mm2 for m in measurements]
        circularities = [m.circularity for m in measurements]
        aspect_ratios = [m.aspect_ratio for m in measurements]

        # --- 图表 1: 粒径直方图 ---
        fig1, ax1 = plt.subplots(figsize=(9, 7)) # 加高画布以容纳说明文字
        ax1.hist(diameters, bins='auto', color='skyblue', edgecolor='black', alpha=0.7)
        ax1.set_title('图1：粒径分布直方图')
        ax1.set_xlabel('等效直径 (mm)')
        ax1.set_ylabel('颗粒数量')
        ax1.grid(axis='y', linestyle='--', alpha=0.5)
        # 添加说明文字
        desc1 = (f"图表说明：展示了不同大小颗粒的数量分布。\n"
                 f"横轴代表颗粒直径，纵轴代表数量。\n"
                 f"可以看到大多数颗粒集中在 {np.mean(diameters):.1f} mm 左右。")
        plt.figtext(0.5, 0.02, desc1, wrap=True, horizontalalignment='center', fontsize=10, color='gray')
        plt.subplots_adjust(bottom=0.15) # 预留底部空间
        plots.append(self._fig_to_numpy(fig1))
        plt.close(fig1)

        # --- 图表 2: 累积分布曲线 ---
        sorted_d = np.sort(diameters)
        y_cum = np.arange(1, len(sorted_d) + 1) / len(sorted_d) * 100
        fig2, ax2 = plt.subplots(figsize=(9, 7))
        ax2.plot(sorted_d, y_cum, marker='.', linestyle='-', color='red', linewidth=2)
        ax2.set_title('图2：粒径累积分布曲线 (CDF)')
        ax2.set_xlabel('等效直径 (mm)')
        ax2.set_ylabel('累积占比 (%)')
        ax2.grid(True, linestyle='--', alpha=0.5)
        # 标注
        for p in [10, 50, 90]:
            d_val = np.percentile(diameters, p)
            ax2.axvline(d_val, color='gray', linestyle=':')
            ax2.text(d_val, p, f' D{p}={d_val:.1f}', color='blue')

        desc2 = ("图表说明：反映了颗粒大小的累积百分比。\n"
                 "D50(中位径)表示有50%的颗粒小于该尺寸。\n"
                 "曲线越陡峭，说明颗粒大小越均匀。")
        plt.figtext(0.5, 0.02, desc2, wrap=True, horizontalalignment='center', fontsize=10, color='gray')
        plt.subplots_adjust(bottom=0.15)
        plots.append(self._fig_to_numpy(fig2))
        plt.close(fig2)

        # --- 图表 3: 形状散点图 ---
        fig3, ax3 = plt.subplots(figsize=(9, 7))
        scatter = ax3.scatter(areas, circularities, c=diameters, cmap='viridis', alpha=0.6)
        ax3.set_title('图3：形状分析 (面积 vs 圆度)')
        ax3.set_xlabel('面积 (mm²)')
        ax3.set_ylabel('圆度 (1.0=正圆)')
        plt.colorbar(scatter, ax=ax3, label='等效直径 (mm)')
        ax3.grid(True, linestyle='--', alpha=0.5)

        desc3 = ("图表说明：展示颗粒形状与大小的关系。\n"
                 "纵轴圆度越接近1，颗粒越圆；越接近0，颗粒越细长。\n"
                 "颜色代表直径大小，黄色越深代表颗粒越大。")
        plt.figtext(0.5, 0.02, desc3, wrap=True, horizontalalignment='center', fontsize=10, color='gray')
        plt.subplots_adjust(bottom=0.15)
        plots.append(self._fig_to_numpy(fig3))
        plt.close(fig3)

        # --- 图表 4: 长宽比箱线图 ---
        fig4, ax4 = plt.subplots(figsize=(9, 7))
        ax4.boxplot(aspect_ratios, vert=False, patch_artist=True,
                    boxprops=dict(facecolor="orange", color="brown"),
                    medianprops=dict(color="black"))
        ax4.set_title('图4：长宽比分布 (Boxplot)')
        ax4.set_xlabel('长宽比 (短轴/长轴)')
        ax4.set_yticks([])
        ax4.set_xlim(0, 1.1) # 强制设置范围0-1
        ax4.grid(axis='x', linestyle='--', alpha=0.5)

        desc4 = ("图表说明：展示颗粒的扁平程度分布。\n"
                 "数值越接近1，说明颗粒越接近圆形或方形；\n"
                 "数值越小，说明颗粒越细长（针状或片状）。\n"
                 "箱体中间的黑线代表中位数。")
        plt.figtext(0.5, 0.02, desc4, wrap=True, horizontalalignment='center', fontsize=10, color='gray')
        plt.subplots_adjust(bottom=0.18)
        plots.append(self._fig_to_numpy(fig4))
        plt.close(fig4)

        return plots

    def _fig_to_numpy(self, fig):
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)