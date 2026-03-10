import sys
import os
import logging

# 1. 配置日志
# 这会让程序在控制台打印出运行过程中的信息（如：加载模型、检测到多少颗粒等）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_environment():
    """
    在启动前检查必要的文件夹和文件是否存在
    """
    # 检查 weights 目录
    if not os.path.exists('weights'):
        try:
            os.makedirs('weights')
            logger.info("已自动创建 'weights' 目录")
        except Exception as e:
            logger.error(f"无法创建 weights 目录: {e}")

    # 检查 SAM 权重文件是否存在
    sam_checkpoint = os.path.join('weights', 'sam_vit_b_01ec64.pth')
    if not os.path.exists(sam_checkpoint):
        logger.warning("=" * 60)
        logger.warning(f"⚠️  未检测到 SAM 模型权重文件: {sam_checkpoint}")
        logger.warning("系统将默认使用 OpenCV 模式。")
        logger.warning("如需使用 AI 模式，请下载模型并放入 weights 文件夹：")
        logger.warning("下载地址: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        logger.warning("=" * 60)
    else:
        logger.info(f"✅ 检测到 SAM 模型权重文件: {sam_checkpoint}")

    # 检查输出目录
    if not os.path.exists('results'):
        os.makedirs('results')


def main():
    """主程序入口"""
    logger.info("正在初始化系统...")

    # 1. 环境检查
    check_environment()

    # 2. 启动 GUI
    try:
        # 延迟导入，这样如果缺少库，可以在这里捕获错误而不是直接报错退出
        from gui.main_window import run_gui

        logger.info("正在启动图形用户界面 (GUI)...")
        run_gui()

    except ImportError as e:
        logger.critical("启动失败：缺少必要的 Python 库！")
        logger.error(f"错误详情: {e}")
        print("\n" + "!" * 50)
        print("请检查是否已安装以下依赖：")
        print("1. PyQt5")
        print("2. opencv-python")
        print("3. torch torchvision")
        print("4. segment-anything (git+https://github.com/facebookresearch/segment-anything.git)")
        print("!" * 50 + "\n")

    except Exception as e:
        logger.critical(f"程序发生未知错误: {e}")
        # 打印详细的错误堆栈，方便调试
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()