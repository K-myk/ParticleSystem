"""
gui/main_window.py
主界面模块 - 仅SAM模式
"""

import sys
import os
import cv2
import logging
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QProgressBar, QTabWidget, QTextEdit, QTableWidget,
    QTableWidgetItem, QFileDialog, QMessageBox, QScrollArea, QFrame,
    QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# --- QSS 样式表 ---
STYLESHEET = """
QMainWindow {
    background-color: #f4f6f9;
}
QGroupBox {
    font-size: 16px;
    font-weight: bold;
    border: 1px solid #dcdcdc;
    border-radius: 8px;
    margin-top: 12px;
    background-color: white;
    padding: 20px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
    color: #2c3e50;
}
QPushButton {
    background-color: #3498db;
    color: white;
    border-radius: 6px;
    padding: 10px 20px;
    font-size: 16px;
    font-weight: bold;
    border: none;
}
QPushButton:hover {
    background-color: #2980b9;
}
QPushButton:disabled {
    background-color: #bdc3c7;
    color: #fff;
}
QPushButton#btnExport {
    background-color: #27ae60;
}
QPushButton#btnExport:hover {
    background-color: #219150;
}
QTabWidget::pane {
    border: 1px solid #cccccc;
    background-color: white;
    border-radius: 6px;
}
QTabBar::tab {
    background: #ecf0f1;
    border: 1px solid #bdc3c7;
    padding: 10px 25px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-size: 15px;
}
QTabBar::tab:selected {
    background: white;
    border-bottom-color: white;
    font-weight: bold;
    color: #2980b9;
}
QProgressBar {
    border: 1px solid #bdc3c7;
    border-radius: 6px;
    text-align: center;
    height: 24px;
    font-size: 14px;
}
QProgressBar::chunk {
    background-color: #1abc9c;
    border-radius: 5px;
}
QLabel {
    color: #34495e;
    font-size: 15px;
}
QTextEdit {
    font-family: "Microsoft YaHei", "Consolas", monospace;
    font-size: 18px;
    line-height: 1.5;
    padding: 15px;
    border: none;
}
QTableWidget {
    font-size: 14px;
}
QHeaderView::section {
    background-color: #ecf0f1;
    padding: 5px;
    font-size: 14px;
    font-weight: bold;
}
"""


class ProcessingThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, image_path, config):
        super().__init__()
        self.image_path = image_path
        self.config = config

    def run(self):
        try:
            from modules import ImagePreprocessor, ParticleDetector, ParticleMeasurer, StatisticalAnalyzer, Visualizer
            from utils import load_image

            self.progress.emit(10, "正在加载图像...")
            image = load_image(self.image_path)
            if image is None: raise Exception("无法加载图像")
            orig_h, orig_w = image.shape[:2]

            self.progress.emit(30, "执行图像预处理...")
            preprocessor = ImagePreprocessor(self.config.preprocess)
            pre_res = preprocessor.process(image)
            proc_h, proc_w = pre_res.final.shape[:2]
            scale_x = orig_w / proc_w
            scale_y = orig_h / proc_h

            self.progress.emit(50, "AI 智能识别中 (SAM模式)...")
            detector = ParticleDetector(self.config.detection)
            det_res = detector.detect(pre_res.final)

            if scale_x != 1.0 or scale_y != 1.0:
                self.progress.emit(60, "还原坐标系...")
                self._rescale_particles(det_res.particles, scale_x, scale_y)

            self.progress.emit(70, "计算颗粒几何参数...")
            measurer = ParticleMeasurer(self.config.measurement)
            meas_res = measurer.measure_batch(det_res.particles, image.shape)

            self.progress.emit(85, "进行统计分析...")
            analyzer = StatisticalAnalyzer()
            stats_res = analyzer.analyze(meas_res.measurements)

            self.progress.emit(90, "生成检测结果图...")
            visualizer = Visualizer(self.config.visualization)
            res_img = visualizer.draw_detection_result(image, det_res.particles, meas_res.measurements)

            self.progress.emit(95, "生成统计图表...")
            plots = visualizer.generate_plots(meas_res.measurements)

            self.finished.emit({
                'result_image': res_img,
                'binary_image': det_res.binary_image,
                'measurements': meas_res.measurements,
                'statistics': stats_res,
                'plots': plots,
                'device_info': detector.get_device_info()
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

    def _rescale_particles(self, particles, scale_x, scale_y):
        for p in particles:
            p.contour = (p.contour.astype(float) * [scale_x, scale_y]).astype(np.int32)
            p.centroid = (p.centroid[0] * scale_x, p.centroid[1] * scale_y)
            p.area = p.area * (scale_x * scale_y)
            p.perimeter = p.perimeter * ((scale_x + scale_y) / 2)
            x, y, w, h = p.bounding_rect
            p.bounding_rect = (int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y))
            p.equivalent_diameter = p.equivalent_diameter * ((scale_x + scale_y) / 2)
            if p.convex_hull is not None:
                p.convex_hull = (p.convex_hull.astype(float) * [scale_x, scale_y]).astype(np.int32)
            p.fitted_ellipse = None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("颗粒物识别系统 (SAM)")
        self.setGeometry(100, 100, 1400, 950)
        self.setStyleSheet(STYLESHEET)

        from config import SystemConfig
        self.config = SystemConfig()
        self.current_image_path = None
        self.current_results = None

        self.display_size = None

        self.setup_ui()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(25)

        # --- 左侧控制栏 ---
        controls = QWidget()
        controls.setFixedWidth(350)
        c_layout = QVBoxLayout(controls)
        c_layout.setContentsMargins(0, 0, 0, 0)
        c_layout.setSpacing(20)

        grp_file = QGroupBox("系统操作")
        f_layout = QVBoxLayout(grp_file)
        f_layout.setSpacing(15)

        self.btn_load = QPushButton("📂 加载图像")
        self.btn_load.setCursor(Qt.PointingHandCursor)
        self.btn_load.clicked.connect(self.load_image)
        f_layout.addWidget(self.btn_load)

        self.btn_process = QPushButton("🚀 开始分析")
        self.btn_process.setCursor(Qt.PointingHandCursor)
        self.btn_process.clicked.connect(self.start_process)
        self.btn_process.setEnabled(False)
        f_layout.addWidget(self.btn_process)

        self.btn_export = QPushButton("💾 导出数据")
        self.btn_export.setObjectName("btnExport")
        self.btn_export.setCursor(Qt.PointingHandCursor)
        self.btn_export.clicked.connect(self.export_data)
        self.btn_export.setEnabled(False)
        f_layout.addWidget(self.btn_export)
        c_layout.addWidget(grp_file)

        grp_params = QGroupBox("识别参数")
        p_layout = QVBoxLayout(grp_params)
        p_layout.setSpacing(15)

        # SAM模式标识（只读标签，替代原来的复选框）
        lbl_mode = QLabel("🤖 检测模式: SAM 深度学习")
        lbl_mode.setStyleSheet("font-size: 15px; color: #27ae60; font-weight: bold;")
        p_layout.addWidget(lbl_mode)

        row1 = QHBoxLayout()
        lbl_min_area = QLabel("最小面积:")
        self.spin_min_area = QSpinBox()
        self.spin_min_area.setRange(10, 10000)
        self.spin_min_area.setValue(50)
        self.spin_min_area.setSuffix(" px")
        self.spin_min_area.setStyleSheet("font-size: 15px; padding: 5px;")
        row1.addWidget(lbl_min_area)
        row1.addWidget(self.spin_min_area)
        p_layout.addLayout(row1)

        row2 = QHBoxLayout()
        lbl_ratio = QLabel("像素比例:")
        self.spin_ratio = QDoubleSpinBox()
        self.spin_ratio.setRange(0.0001, 10.0)
        self.spin_ratio.setValue(0.1)
        self.spin_ratio.setDecimals(4)
        self.spin_ratio.setSuffix(" mm/px")
        self.spin_ratio.setStyleSheet("font-size: 15px; padding: 5px;")
        row2.addWidget(lbl_ratio)
        row2.addWidget(self.spin_ratio)
        p_layout.addLayout(row2)

        c_layout.addWidget(grp_params)

        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        c_layout.addWidget(self.progress)

        self.lbl_status = QLabel("等待操作...")
        self.lbl_status.setStyleSheet("color: #7f8c8d; font-style: italic; margin-top: 5px; font-size: 14px;")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        c_layout.addWidget(self.lbl_status)

        c_layout.addStretch()
        layout.addWidget(controls)

        # --- 右侧显示区 ---
        self.tabs = QTabWidget()

        self.lbl_img_result = QLabel("结果预览")
        self.lbl_img_result.setAlignment(Qt.AlignCenter)
        self.lbl_img_result.setStyleSheet("background-color: #ecf0f1; border: 2px dashed #bdc3c7;")
        self.tabs.addTab(self.lbl_img_result, "👁️ 检测结果")

        self.lbl_img_binary = QLabel("二值化预览")
        self.lbl_img_binary.setAlignment(Qt.AlignCenter)
        self.lbl_img_binary.setStyleSheet("background-color: black; border: none;")
        self.tabs.addTab(self.lbl_img_binary, "⚫ 二值化")

        self.scroll_plots = QScrollArea()
        self.scroll_plots.setWidgetResizable(True)
        self.plot_container = QWidget()
        self.plot_container.setStyleSheet("background-color: white;")
        self.plot_layout = QVBoxLayout(self.plot_container)
        self.plot_layout.setSpacing(40)
        self.plot_layout.setContentsMargins(40, 40, 40, 40)
        self.scroll_plots.setWidget(self.plot_container)
        self.tabs.addTab(self.scroll_plots, "📊 数据图表")

        self.table_data = QTableWidget()
        self.table_data.setAlternatingRowColors(True)
        self.tabs.addTab(self.table_data, "📋 详细数据")

        self.txt_stats = QTextEdit()
        self.txt_stats.setReadOnly(True)
        self.tabs.addTab(self.txt_stats, "📝 统计报告")

        layout.addWidget(self.tabs)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "Images (*.jpg *.png *.bmp)")
        if path:
            self.current_image_path = path
            self.btn_process.setEnabled(True)
            self.btn_export.setEnabled(False)
            self.lbl_status.setText(f"已加载: {os.path.basename(path)}")

            self.display_size = None

            pix = QPixmap(path)
            self.lbl_img_result.setPixmap(pix.scaled(self.lbl_img_result.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.lbl_img_binary.clear()
            self.clear_plots()

    def clear_plots(self):
        while self.plot_layout.count():
            item = self.plot_layout.takeAt(0)
            widget = item.widget()
            if widget: widget.deleteLater()

    def start_process(self):
        if not self.current_image_path: return
        self.config.detection.min_area = self.spin_min_area.value()
        self.config.measurement.pixel_to_mm = self.spin_ratio.value()
        self.btn_process.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.thread = ProcessingThread(self.current_image_path, self.config)
        self.thread.progress.connect(self.update_progress)
        self.thread.finished.connect(self.process_finished)
        self.thread.error.connect(self.process_error)
        self.thread.start()

    def update_progress(self, val, msg):
        self.progress.setValue(val)
        self.lbl_status.setText(msg)

    def process_finished(self, res):
        self.current_results = res
        self.btn_process.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.progress.setValue(100)
        self.lbl_status.setText(f"✅ 处理完成 | {res['device_info']}")

        # 1. 显示检测结果图
        h, w, c = res['result_image'].shape
        rgb_image = cv2.cvtColor(res['result_image'], cv2.COLOR_BGR2RGB).copy()
        qimg = QImage(rgb_image.data, w, h, w*c, QImage.Format_RGB888)
        result_pixmap = QPixmap.fromImage(qimg).scaled(
            self.lbl_img_result.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.lbl_img_result.setPixmap(result_pixmap)

        self.display_size = result_pixmap.size()

        # 2. 显示二值化图（使用与检测结果图完全相同的尺寸）
        h, w = res['binary_image'].shape
        bin_image = res['binary_image'].copy()
        qimg_bin = QImage(bin_image.data, w, h, w, QImage.Format_Grayscale8)
        binary_pixmap = QPixmap.fromImage(qimg_bin).scaled(
            self.display_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.lbl_img_binary.setPixmap(binary_pixmap)

        # 3. 显示统计报告
        from modules import StatisticalAnalyzer
        analyzer = StatisticalAnalyzer()
        self.txt_stats.setText(analyzer.generate_report(res['statistics']))

        # 4. 填充表格
        self.fill_table(res['measurements'])

        # 5. 显示图表
        self.show_plots(res['plots'])

        self.tabs.setCurrentIndex(4)

    def fill_table(self, measurements):
        self.table_data.setRowCount(len(measurements))
        self.table_data.setColumnCount(6)
        self.table_data.setHorizontalHeaderLabels(["ID", "面积(mm²)", "周长(mm)", "等效直径(mm)", "圆度", "长宽比"])
        for i, m in enumerate(measurements):
            self.table_data.setItem(i, 0, QTableWidgetItem(str(m.particle_id)))
            self.table_data.setItem(i, 1, QTableWidgetItem(f"{m.area_mm2:.2f}"))
            self.table_data.setItem(i, 2, QTableWidgetItem(f"{m.perimeter_mm:.2f}"))
            self.table_data.setItem(i, 3, QTableWidgetItem(f"{m.equivalent_diameter_mm:.2f}"))
            self.table_data.setItem(i, 4, QTableWidgetItem(f"{m.circularity:.3f}"))
            self.table_data.setItem(i, 5, QTableWidgetItem(f"{m.aspect_ratio:.2f}"))

    def show_plots(self, plots):
        self.clear_plots()
        if not plots: return
        for plot_img in plots:
            h, w, c = plot_img.shape
            rgb_plot = cv2.cvtColor(plot_img, cv2.COLOR_BGR2RGB).copy()
            qimg_plot = QImage(rgb_plot.data, w, h, w*c, QImage.Format_RGB888)
            lbl_plot = QLabel()
            lbl_plot.setPixmap(QPixmap.fromImage(qimg_plot))
            lbl_plot.setAlignment(Qt.AlignCenter)
            lbl_plot.setStyleSheet("border: 1px solid #ddd; margin-bottom: 20px;")
            self.plot_layout.addWidget(lbl_plot)

    def export_data(self):
        if not self.current_results: return

        save_dir_root = QFileDialog.getExistingDirectory(self, "选择保存根目录")
        if not save_dir_root: return

        try:
            from utils import save_results_to_excel, save_image, generate_timestamp
            from modules import ParticleMeasurer, StatisticalAnalyzer

            timestamp = generate_timestamp()

            export_folder_name = f"Result_{timestamp}"
            export_dir = os.path.join(save_dir_root, export_folder_name)
            os.makedirs(export_dir, exist_ok=True)

            class DummyBatchResult:
                def __init__(self, ms): self.measurements = ms
            measurer = ParticleMeasurer(self.config.measurement)
            measurements_data = measurer.export_measurements(DummyBatchResult(self.current_results['measurements']))
            analyzer = StatisticalAnalyzer()
            stats_dict = analyzer.export_to_dict(self.current_results['statistics'])

            excel_name = f"Data_{timestamp}.xlsx"
            save_results_to_excel(measurements_data, stats_dict, os.path.join(export_dir, excel_name))

            save_image(self.current_results['result_image'], os.path.join(export_dir, f"Img_Detection_{timestamp}.png"))
            save_image(self.current_results['binary_image'], os.path.join(export_dir, f"Img_Binary_{timestamp}.png"))

            plot_names = ["Chart_Histogram", "Chart_CDF", "Chart_Scatter", "Chart_Boxplot"]
            for i, plot_img in enumerate(self.current_results['plots']):
                if i < len(plot_names):
                    save_image(plot_img, os.path.join(export_dir, f"{plot_names[i]}_{timestamp}.png"))

            QMessageBox.information(self, "导出成功", f"所有文件已归档到文件夹:\n{export_dir}")

        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"发生错误: {str(e)}")
            logger.error(f"导出失败: {e}")

    def process_error(self, msg):
        self.btn_process.setEnabled(True)
        self.btn_export.setEnabled(False)
        QMessageBox.critical(self, "错误", msg)
        self.lbl_status.setText("发生错误")


def run_gui():
    app = QApplication(sys.argv)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())