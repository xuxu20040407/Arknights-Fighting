from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QRubberBand, QApplication, QSizePolicy
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QImage
from PyQt6.QtCore import Qt, QRect, QPoint, QSize, pyqtSignal
import numpy as np
import cv2
import sys

from src.core.log import logger # Import the logger

class ImageViewer(QWidget):
    """
    A widget to display an image (screenshot) and allow the user to select
    a rectangular area on it using a rubber band.
    Emits a signal 'new_selection(QRect)' when a new area is selected.
    """
    new_selection = pyqtSignal(QRect) # Signal emitting the selected QRect

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._pixmap = QPixmap()
        # Remove QLabel, we will draw directly in paintEvent
        # self.image_label = QLabel(self)
        # self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # self.image_label.setScaledContents(False) # Disable auto-scaling

        # layout = QVBoxLayout(self)
        # layout.addWidget(self.image_label) # Remove label from layout
        # layout.setContentsMargins(0, 0, 0, 0)

        self.rubber_band = None
        self.origin = QPoint()
        self._selection_rect = QRect() # Stores the current selection relative to the widget
        self._image_selection_rect = QRect() # Stores the selection relative to the original image

        self.original_image_size = QSize(0, 0) # Store the original image dimensions

    def set_image(self, image_np: np.ndarray):
        """Sets the image to be displayed from a NumPy array (BGR format)."""
        if image_np is None or image_np.size == 0:
            self._pixmap = QPixmap() # Clear pixmap
            # self.image_label.clear() # No label to clear
            self.original_image_size = QSize(0, 0)
            self._selection_rect = QRect()
            self._image_selection_rect = QRect()
            self.update() # Trigger repaint to clear the view
            return

        height, width, channel = image_np.shape
        bytes_per_line = 3 * width
        q_image = QImage(image_np.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)

        self.original_image_size = q_image.size()
        self._pixmap = QPixmap.fromImage(q_image)

        # Don't set pixmap on label anymore
        # self.image_label.setPixmap(...)

        self._selection_rect = QRect() # Clear previous selection on new image
        self._image_selection_rect = QRect()
        self.update() # Trigger repaint to show the new image

    def resizeEvent(self, event):
        """Handle widget resize to rescale the displayed pixmap."""
        # Just trigger a repaint, paintEvent handles scaling
        super().resizeEvent(event)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and not self._pixmap.isNull():
            # Clear previous selection and redraw to remove red box
            self._selection_rect = QRect()
            self._image_selection_rect = QRect()
            self.update()

            self.origin = event.pos() # Position relative to the widget
            if not self.rubber_band:
                self.rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)

            # Check if click is within the drawable area (widget bounds)
            # We'll check against pixmap bounds during move/release
            if self.rect().contains(self.origin):
                 # Start rubber band from the click position
                 self.rubber_band.setGeometry(QRect(self.origin, QSize()))
                 self.rubber_band.show()
            else:
                self.origin = QPoint() # Reset origin if click is outside widget

    def mouseMoveEvent(self, event):
        if self.rubber_band and not self.origin.isNull():
            # Constrain rubber band within the widget bounds (or displayed pixmap bounds)
            # Clamp the rubber band to the widget's boundaries
            end_point = event.pos()
            clamped_x = max(0, min(end_point.x(), self.width()))
            clamped_y = max(0, min(end_point.y(), self.height()))
            clamped_end_point = QPoint(clamped_x, clamped_y)
            self.rubber_band.setGeometry(QRect(self.origin, clamped_end_point).normalized())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.rubber_band and not self.origin.isNull():
            if self.rubber_band.isVisible():
                 self.rubber_band.hide()
                 # Get the final geometry in widget coordinates
                 widget_selection = self.rubber_band.geometry().normalized()

                 # Only proceed if the selection has a valid size
                 if widget_selection.width() > 0 and widget_selection.height() > 0:
                      # Intersect with the actual displayed pixmap bounds before mapping
                      pixmap_rect = self._get_displayed_pixmap_rect()
                      valid_selection_widget = widget_selection.intersected(pixmap_rect)

                      if valid_selection_widget.isValid() and valid_selection_widget.width() > 0 and valid_selection_widget.height() > 0:
                           self._selection_rect = valid_selection_widget # Store valid selection in widget coords for drawing red box
                           # Map the valid widget selection rect to original image coordinates
                           self._image_selection_rect = self._map_widget_rect_to_image_rect(self._selection_rect)

                           logger.debug(f"Widget Selection Rect (Final): {self._selection_rect}")
                           logger.debug(f"Image Selection Rect (Mapped): {self._image_selection_rect}")

                           if self._image_selection_rect.isValid() and self._image_selection_rect.width() > 0 and self._image_selection_rect.height() > 0:
                                self.new_selection.emit(self._image_selection_rect) # Emit signal with image coords
                           else:
                                self.new_selection.emit(QRect()) # Emit invalid if mapping failed
                                self._selection_rect = QRect() # Clear widget selection too
                      else:
                           # Selection was outside or invalid after intersection
                           self._selection_rect = QRect()
                           self._image_selection_rect = QRect()
                           self.new_selection.emit(QRect())
                 else:
                      # Selection width/height was zero
                      self._selection_rect = QRect()
                      self._image_selection_rect = QRect()
                      self.new_selection.emit(QRect())

                 self.update() # Trigger repaint to draw the red box (or clear it)

            self.origin = QPoint() # Reset origin after selection

    def get_selection(self) -> QRect:
        """Returns the selected rectangle in original image coordinates."""
        return self._image_selection_rect

    def _get_displayed_pixmap_rect(self) -> QRect:
        """Calculates the rectangle where the scaled pixmap is drawn within the widget."""
        if self._pixmap.isNull():
            return QRect()

        # Get the scaled size based on widget size and aspect ratio
        scaled_pixmap = self._pixmap.scaled(
            self.size(), # Scale to fit the widget size
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        pixmap_size = scaled_pixmap.size()
        widget_size = self.size()

        # Calculate top-left corner for centering
        x_offset = (widget_size.width() - pixmap_size.width()) // 2
        y_offset = (widget_size.height() - pixmap_size.height()) // 2

        return QRect(x_offset, y_offset, pixmap_size.width(), pixmap_size.height())

    def _map_widget_rect_to_image_rect(self, widget_rect: QRect) -> QRect:
        """Maps a rectangle from widget coordinates to original image coordinates."""
        if self._pixmap.isNull() or self.original_image_size.isEmpty():
            return QRect()

        displayed_rect = self._get_displayed_pixmap_rect()
        if not displayed_rect.isValid():
            return QRect()

        # Normalize the widget_rect relative to the displayed pixmap area
        relative_x = widget_rect.x() - displayed_rect.x()
        relative_y = widget_rect.y() - displayed_rect.y()
        relative_width = widget_rect.width()
        relative_height = widget_rect.height()

        # Calculate scaling factors
        scale_x = self.original_image_size.width() / displayed_rect.width()
        scale_y = self.original_image_size.height() / displayed_rect.height()

        # Map to original image coordinates
        image_x = int(relative_x * scale_x)
        image_y = int(relative_y * scale_y)
        image_width = int(relative_width * scale_x)
        image_height = int(relative_height * scale_y)

        # Clamp to original image bounds to avoid minor rounding errors going out of bounds
        image_x = max(0, min(image_x, self.original_image_size.width()))
        image_y = max(0, min(image_y, self.original_image_size.height()))
        image_width = max(0, min(image_width, self.original_image_size.width() - image_x))
        image_height = max(0, min(image_height, self.original_image_size.height() - image_y))


        return QRect(image_x, image_y, image_width, image_height)


    def paintEvent(self, event):
        """Handles painting the scaled image and the selection rectangle."""
        super().paintEvent(event) # Call parent's paintEvent if needed
        painter = QPainter(self)

        if not self._pixmap.isNull():
            # Calculate the rectangle to draw the pixmap centered and scaled
            target_rect = self._get_displayed_pixmap_rect()
            # Draw the scaled pixmap
            painter.drawPixmap(target_rect, self._pixmap)

            # Draw the red selection rectangle if it's valid
            if self._selection_rect.isValid() and self._selection_rect.width() > 0 and self._selection_rect.height() > 0:
                pen = QPen(QColor('red'), 2, Qt.PenStyle.SolidLine) # 2px solid red line
                painter.setPen(pen)
                # Draw rectangle based on widget coordinates stored in _selection_rect
                painter.drawRect(self._selection_rect)
        else:
             # Optionally paint a background or placeholder if no image
             painter.fillRect(self.rect(), QColor(Qt.GlobalColor.darkGray))
             painter.setPen(QColor(Qt.GlobalColor.white))
             painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "无图像")


# Example usage for testing the widget standalone
if __name__ == '__main__':
    app = QApplication([])

    # Create a dummy numpy image (e.g., gradient)
    dummy_img_np = np.zeros((300, 400, 3), dtype=np.uint8)
    dummy_img_np[:, :, 0] = np.tile(np.linspace(0, 255, 400), (300, 1)) # Blue gradient
    dummy_img_np[:, :, 1] = np.tile(np.linspace(0, 255, 300), (400, 1)).T # Green gradient

    viewer = ImageViewer()
    viewer.set_image(dummy_img_np)

    def handle_selection(rect):
        logger.debug(f"Signal Received: New selection in image coordinates: {rect}")
        # You could potentially crop the original image here using the rect
        if rect.isValid() and dummy_img_np is not None:
             cropped = dummy_img_np[rect.y():rect.y()+rect.height(), rect.x():rect.x()+rect.width()]
             if cropped.size > 0:
                 try:
                     cv2.imwrite("cropped_test.png", cropped)
                     print("Saved cropped area to cropped_test.png")
                 except Exception as e:
                     print(f"Could not save cropped image: {e}")


    viewer.new_selection.connect(handle_selection)
    viewer.resize(600, 450) # Give it a size
    viewer.show()

    sys.exit(app.exec())