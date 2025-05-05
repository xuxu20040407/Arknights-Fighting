import platform  # 导入 platform 模块
from PyQt6.QtCore import Qt, QRect, QPoint, pyqtSignal, QSize
from PyQt6.QtGui import QPainter, QPen, QColor, QScreen
from PyQt6.QtWidgets import QApplication, QWidget, QRubberBand

from src.core.log import logger  # 导入日志记录器


class ScreenSelectionWidget(QWidget):
    """
    一个覆盖屏幕的透明窗口，允许用户通过拖拽选择一个矩形区域。
    选择完成后发出 'area_selected(QRect)' 信号，包含所选区域的屏幕坐标。
    """
    area_selected = pyqtSignal(QRect)

    def __init__(self, screen: QScreen):
        super().__init__()
        self.screen_geometry = screen.geometry()
        self.setGeometry(self.screen_geometry)
        self.setWindowTitle('选择识别区域')
        # 设置窗口标志：无边框、保持最前、半透明
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        raw_pixel_ratio = screen.devicePixelRatio()
        if platform.system() == 'Darwin': 
            self.device_pixel_ratio = raw_pixel_ratio / 2.0
        else:
            self.device_pixel_ratio = raw_pixel_ratio
        self.rubber_band = None
        self.origin = QPoint()
        self.current_pos = QPoint() # 存储鼠标移动或释放时的全局坐标
        self.preview_rect = QRect() # 存储鼠标释放后的预览矩形

        # 添加一个半透明的黑色背景以便看清选区
        self.overlay_color = QColor(0, 0, 0, 100) # 半透明黑色

    def paintEvent(self, event):
        """绘制半透明背景和预览选框"""
        painter = QPainter(self)
        # 绘制半透明背景
        painter.fillRect(self.rect(), self.overlay_color)

        # 如果有预览选区 (全局坐标)，将其转换为局部坐标进行绘制
        if self.preview_rect.isValid():
            # 将全局 preview_rect 转换为相对于当前窗口的局部坐标
            # 注意：self.geometry() 返回的是相对于父窗口的几何，对于顶级窗口（如这里），
            # 其 topLeft() 通常是 (0,0)，除非它被移动过。
            # 但为了健壮性，我们假设窗口可能不在 (0,0) 开始（虽然全屏窗口通常是这样）。
            # ScreenSelectionWidget 的 geometry 是全屏的，其 topLeft() 就是屏幕左上角。
            # 因此，从全局坐标转换到窗口局部坐标，需要减去窗口的全局左上角坐标。
            draw_rect = self.preview_rect.translated(-self.geometry().topLeft())
            pen = QPen(QColor('red'), 2, Qt.PenStyle.SolidLine) # 2px 红色实线
            painter.setPen(pen)
            painter.drawRect(draw_rect) # 绘制局部坐标矩形

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # 清除之前的预览框
            self.preview_rect = QRect()
            self.update() # 触发重绘以清除红框

            self.origin = event.globalPosition().toPoint() # 记录全局逻辑坐标起点
            logger.debug(f"Debug: mousePress - Global Logical Origin: {self.origin}") # 调试打印
            if not self.rubber_band:
                self.rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)

            # 将全局起点映射到窗口局部坐标，用于设置橡皮筋初始位置
            local_origin = self.mapFromGlobal(self.origin)
            logger.debug(f"Debug: mousePress - Mapped Local Origin: {local_origin}") # 调试打印
            self.rubber_band.setGeometry(QRect(local_origin, QSize())) # 使用局部坐标
            self.rubber_band.show()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.rubber_band and not self.origin.isNull() and event.buttons() & Qt.MouseButton.LeftButton:
            self.current_pos = event.globalPosition().toPoint() # 记录当前全局逻辑坐标
            # print(f"Debug: mouseMove - Global Logical Current: {self.current_pos}") # 频繁打印，可选

            # 将全局起点和当前点映射到窗口局部坐标，用于更新橡皮筋
            local_origin = self.mapFromGlobal(self.origin)
            local_current = self.mapFromGlobal(self.current_pos)
            # print(f"Debug: mouseMove - Mapped Local Origin: {local_origin}, Mapped Local Current: {local_current}") # 频繁打印，可选
            self.rubber_band.setGeometry(QRect(local_origin, local_current).normalized()) # 使用局部坐标
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.rubber_band:
            self.current_pos = event.globalPosition().toPoint() # 记录释放时的全局逻辑坐标
            logger.debug(f"Debug: mouseRelease - Global Logical Release Pos: {self.current_pos}") # 调试打印
            self.rubber_band.hide() # 隐藏橡皮筋

            # 直接使用全局起点和全局释放点计算最终的全局逻辑坐标矩形
            self.preview_rect = QRect(self.origin, self.current_pos).normalized()
            logger.debug(f"Debug: mouseRelease - Calculated Global Logical Rect: {self.preview_rect}") # 调试打印

            # 检查计算出的全局选区是否有效
            if self.preview_rect.isValid() and self.preview_rect.width() > 0 and self.preview_rect.height() > 0:
                logger.info(f"预览选区已确定 (全局逻辑坐标): {self.preview_rect}. 按 Enter 确认, Esc 取消, 或重新拖拽。") # 更新提示
            else:
                logger.warning("拖拽选区无效。")
                self.preview_rect = QRect() # 重置预览

            self.update() # 触发重绘以显示（或清除）红框
            # 不需要重置 self.origin，以便用户可以重新拖拽
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        key = event.key()
        # 按 Enter 键确认当前预览选区
        if key == Qt.Key.Key_Return or key == Qt.Key.Key_Enter:
            if self.preview_rect.isValid():
                # 确认发射的是最终计算出的全局逻辑坐标
                logger.info(f"选区已通过 Enter 确认 (全局逻辑坐标): {self.preview_rect}")

                # 将逻辑坐标转换为物理像素坐标
                physical_rect = QRect(
                    int(self.preview_rect.left() * self.device_pixel_ratio),
                    int(self.preview_rect.top() * self.device_pixel_ratio),
                    int(self.preview_rect.width() * self.device_pixel_ratio),
                    int(self.preview_rect.height() * self.device_pixel_ratio)
                )
                logger.debug(f"Debug: keyPress - Calculated Physical Rect (for emission): {physical_rect}") # 调试打印

                # 发射物理像素坐标
                self.area_selected.emit(physical_rect)
                self.close()
                event.accept()
            else:
                logger.warning("没有有效的预览选区可供确认。")
                event.ignore() # 没有有效选区，忽略 Enter

        # 按 Esc 键取消选择并关闭窗口
        elif key == Qt.Key.Key_Escape:
            logger.info("选区被 Esc 取消。")
            self.area_selected.emit(QRect()) # 发出空矩形信号
            self.close()
            event.accept()
        else:
            super().keyPressEvent(event)

# 注意：不再需要 capture_full_screen 函数

if __name__ == '__main__':
    # 测试 ScreenSelectionWidget
    # 显式启用 High DPI 缩放支持 (使用正确的枚举路径)
    try:
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
    except AttributeError:
        logger.warning("警告: 无法设置 AA_EnableHighDpiScaling (可能在您的 Qt 版本中不需要或位置不同)。")
    app = QApplication([])
    primary_screen = QApplication.primaryScreen()
    if primary_screen:
        # 打印设备像素比例 (现在在 __init__ 中获取)
        # dpi_ratio = primary_screen.devicePixelRatio()
        # print(f"测试: 主屏幕设备像素比例 (缩放因子): {dpi_ratio}")
        selector = ScreenSelectionWidget(primary_screen)
        logger.debug(f"测试: 主屏幕设备像素比例 (缩放因子): {selector.device_pixel_ratio}") # 从实例获取
        def selection_done(rect):
            if rect.isValid():
                logger.debug(f"测试: 收到选区信号 (物理像素坐标): {rect}") # 更新提示
            else:
                logger.debug("测试: 收到无效选区信号。")
            app.quit() # 收到信号后退出测试

        selector.area_selected.connect(selection_done)
        logger.debug("测试: 显示屏幕选择窗口，请拖拽选择一个区域或按 Esc 取消。")
        selector.show()
        app.exec()
    else:
        logger.debug("测试: 无法获取主屏幕。")