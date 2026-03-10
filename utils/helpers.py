"""
utils/helpers.py
辅助函数模块：包含图像读写、Excel/JSON导出、时间戳生成等
"""

import os
import cv2
import numpy as np
import json
import time
import logging
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

# 设置日志
logger = logging.getLogger(__name__)


class Timer:
    """计时器上下文管理器"""

    def __init__(self, name="Task"):
        self.name = name
        self.start = 0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        logger.info(f"[{self.name}] 耗时: {elapsed:.4f}秒")


def get_image_files(directory: str, extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')) -> List[
    str]:
    """获取目录下所有图像文件"""
    image_files = []
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if filename.lower().endswith(extensions):
                image_files.append(os.path.join(directory, filename))
    return sorted(image_files)


def load_image(path: str) -> np.ndarray:
    """加载图像文件 (支持中文路径)"""
    if not os.path.exists(path):
        logger.error(f"文件不存在: {path}")
        return None
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"读取图像失败: {e}")
        return None


def save_image(image: np.ndarray, path: str):
    """保存图像文件 (支持中文路径)"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imencode(os.path.splitext(path)[1], image)[1].tofile(path)
    except Exception as e:
        logger.error(f"保存图像失败: {e}")


def generate_timestamp() -> str:
    """生成当前时间戳字符串 (用于文件名)"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_results_to_json(data: Any, path: str):
    """保存数据到 JSON 文件"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"JSON保存成功: {path}")
    except Exception as e:
        logger.error(f"JSON保存失败: {e}")


def save_results_to_excel(measurements: list, stats: dict, path: str) -> bool:
    """保存测量数据和统计结果到 Excel (多Sheet模式)"""
    try:
        import pandas as pd
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 创建 Excel Writer
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            # Sheet 1: 详细测量数据
            df_measurements = pd.DataFrame(measurements)
            df_measurements.to_excel(writer, sheet_name='详细数据', index=False)

            # Sheet 2: 统计摘要
            # 将字典转换为 DataFrame (键值对)
            stats_list = [{"统计项": k, "数值": v} for k, v in stats.items()]
            df_stats = pd.DataFrame(stats_list)
            df_stats.to_excel(writer, sheet_name='统计报告', index=False)

        logger.info(f"Excel保存成功: {path}")
        return True
    except ImportError:
        logger.error("保存Excel失败: 未安装 pandas 或 openpyxl 库，请使用 pip install pandas openpyxl 安装")
        return False
    except Exception as e:
        logger.error(f"保存Excel失败: {e}")
        return False


def load_results_from_json(filepath: str) -> Optional[Dict[str, Any]]:
    """从JSON文件加载结果"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载JSON失败: {e}")
        return None


def resize_image_to_fit(image: np.ndarray, max_width: int = 1200, max_height: int = 800) -> np.ndarray:
    """调整图像大小以适应显示"""
    h, w = image.shape[:2]
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h, 1.0)

    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image


def calculate_pixel_to_real_ratio(reference_pixels: float, reference_mm: float) -> float:
    """计算像素比例"""
    if reference_pixels <= 0: return 0.1
    return reference_mm / reference_pixels