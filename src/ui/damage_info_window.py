import math
import os
from typing import List, TYPE_CHECKING

from PyQt6.QtCore import Qt, QPoint, QRectF  # Use QRectF for floating point precision if needed
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QTextDocument, QTextOption  # Import QTextDocument
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QWidget, QScrollArea,
    QSizePolicy  # Import QMessageBox here for error handling
)

# Adjust import path for data directory
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
TEMPLATE_DIR = os.path.join(DATA_DIR, 'image')

# Use TYPE_CHECKING to avoid circular import issues
if TYPE_CHECKING:
    from src.models.monster import Monster
    from src.core.calculator import calculate_damage

# Import after TYPE_CHECKING block
from src.models.monster import Monster
from src.core.calculator import calculate_damage


class DamageArrowWidget(QWidget):
    """A widget to draw an arrow (left OR right) with DPH and DPS+% text near the tail."""
    # Accepts dph_text (now including percent) and dps_text
    def __init__(self, dph_percent_text: str, dps_text: str, direction: str, tooltip: str = "", parent=None):
        super().__init__(parent)
        self.dph_percent_text = dph_percent_text # DPH and Percent text
        self.dps_text = dps_text                 # DPS text only
        self.direction = direction # 'left' or 'right'
        self.setToolTip(tooltip)
        self.setMinimumHeight(45) # Slightly reduce height for potentially shorter text lines
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.arrow_margin = 15 # Margin from edge to start/end of arrow line
        self.text_margin = 5   # Margin from edge to text block
        self.text_width_ratio = 0.6 # How much width the text block can take

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pen = QPen(QColor("black"))
        pen.setWidth(2)
        painter.setPen(pen)

        arrow_y = self.height() // 2
        arrow_size = 8 # Smaller arrowhead

        if self.direction == 'right': # Arrow points right ->
            start_point = QPoint(self.arrow_margin, arrow_y)
            end_point = QPoint(self.width() - self.arrow_margin, arrow_y)
            painter.drawLine(start_point, end_point)
            # Arrowhead at right end
            angle = 0 # Horizontal line
            arrow_p1 = end_point + QPoint(int(-arrow_size * math.cos(angle - math.pi / 6)), int(-arrow_size * math.sin(angle - math.pi / 6)))
            arrow_p2 = end_point + QPoint(int(-arrow_size * math.cos(angle + math.pi / 6)), int(-arrow_size * math.sin(angle + math.pi / 6)))
            painter.drawLine(end_point, arrow_p1)
            painter.drawLine(end_point, arrow_p2)

            # Text near left tail
            text_x = self.text_margin
            text_w = int(self.width() * self.text_width_ratio)
            text_align = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter

        elif self.direction == 'left': # Arrow points left <-
            start_point = QPoint(self.width() - self.arrow_margin, arrow_y)
            end_point = QPoint(self.arrow_margin, arrow_y)
            painter.drawLine(start_point, end_point)
            # Arrowhead at left end
            angle = math.pi # Horizontal line pointing left
            arrow_p1 = end_point + QPoint(int(-arrow_size * math.cos(angle - math.pi / 6)), int(-arrow_size * math.sin(angle - math.pi / 6)))
            arrow_p2 = end_point + QPoint(int(-arrow_size * math.cos(angle + math.pi / 6)), int(-arrow_size * math.sin(angle + math.pi / 6)))
            painter.drawLine(end_point, arrow_p1)
            painter.drawLine(end_point, arrow_p2)

            # Text near right tail
            text_w = int(self.width() * self.text_width_ratio)
            text_x = self.width() - text_w - self.text_margin
            text_align = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        else:
            return # Invalid direction

        # --- Draw Rich Text using QTextDocument ---
        # DPH Text
        dph_doc = QTextDocument()
        dph_doc.setDefaultFont(QFont("Arial", 9)) # Set default font
        # Set alignment within the document itself
        option = QTextOption(text_align) # Use alignment derived earlier
        option.setWrapMode(QTextOption.WrapMode.NoWrap) # Prevent wrapping
        dph_doc.setDefaultTextOption(option)
        dph_doc.setHtml(self.dph_percent_text) # Use the correct attribute name
        dph_doc.setTextWidth(text_w) # Constrain width

        # Calculate vertical position for DPH text (above arrow)
        # Use sizeHint for better height calculation with rich text
        dph_text_height = dph_doc.size().height()
        dph_rect = QRectF(text_x, arrow_y - dph_text_height - 1, text_w, dph_text_height)

        painter.save()
        painter.translate(dph_rect.topLeft())
        dph_doc.drawContents(painter)
        painter.restore()

        # Combined DPS + Percent Text
        dps_percent_doc = QTextDocument()
        dps_percent_doc.setDefaultFont(QFont("Arial", 9))
        dps_percent_doc.setDefaultTextOption(option) # Reuse alignment option
        dps_percent_doc.setHtml(self.dps_text) # Set DPS text only
        dps_percent_doc.setTextWidth(text_w)

        # Calculate vertical position for DPS text (below arrow)
        dps_text_height = dps_percent_doc.size().height()
        dps_rect = QRectF(text_x, arrow_y + 1, text_w, dps_text_height) # Renamed variable

        painter.save()
        painter.translate(dps_rect.topLeft()) # Use renamed variable
        dps_percent_doc.drawContents(painter)
        painter.restore()


class DamageInfoWindow(QDialog):
    def __init__(self, left_monsters: List['Monster'], right_monsters: List['Monster'],
                 left_side_name: str, right_side_name: str, parent=None):
        super().__init__(parent)
        self.left_monsters = left_monsters
        self.right_monsters = right_monsters
        self.left_side_name = left_side_name
        self.right_side_name = right_side_name

        # Determine which monster was clicked (focused) and which list is the target
        if len(self.left_monsters) == 1 and len(self.right_monsters) >= 1:
            self.clicked_monster = self.left_monsters[0]
            self.other_monsters = self.right_monsters
            self.clicked_side_name = left_side_name
            self.other_side_name = right_side_name
        elif len(self.right_monsters) == 1 and len(self.left_monsters) >= 1:
            self.clicked_monster = self.right_monsters[0]
            self.other_monsters = self.left_monsters
            self.clicked_side_name = right_side_name
            self.other_side_name = left_side_name
        else:
            QMessageBox.critical(self, "错误", "输入数据无效。左侧或右侧应只有一个怪物被选中。")
            self.clicked_monster = None
            self.other_monsters = []
            self.reject()
            return

        if not self.clicked_monster:
             QMessageBox.critical(self, "错误", "未能加载选中的怪物信息。")
             self.reject()
             return

        self.setWindowTitle(f"{self.clicked_monster.name} vs {self.other_side_name} - 双向伤害")
        self.setGeometry(150, 150, 750, 500) # Reduced size

        # Main layout is just the scroll area now
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5) # Reduced margins
        self.main_layout.setSpacing(0) # No spacing, rows manage their own

        # --- Scroll Area ---
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; }") # Remove scroll area border
        self.scroll_content_widget = QWidget()
        # This layout holds the horizontal rows
        self.rows_layout = QVBoxLayout(self.scroll_content_widget)
        self.rows_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.rows_layout.setSpacing(5) # Reduced spacing between rows
        self.rows_layout.setContentsMargins(3, 3, 3, 3) # Reduced margins

        self.scroll_area.setWidget(self.scroll_content_widget)
        self.main_layout.addWidget(self.scroll_area) # Scroll area takes all space

        self._populate_damage_rows()

    def _create_monster_display(self, monster: 'Monster', size: int = 80) -> QWidget:
        """Creates a simple widget to display a monster's image and name."""
        widget = QWidget()
        # Make background transparent, remove border for use within rows
        widget.setStyleSheet("background-color: transparent; border: none;")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(2) # Reduced spacing
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Image Label
        # Image Label
        image_label = QLabel()
        image_label.setFixedSize(size, size) # Use passed size
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;") # Keep border for image itself

        image_path = os.path.join(TEMPLATE_DIR, f"{monster.template_name}.png")
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            image_label.setText("无图")
        else:
            image_label.setPixmap(pixmap.scaled(image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        # Name Label
        name_label = QLabel(f"<b>{monster.name}</b>")
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_label.setWordWrap(True)

        layout.addWidget(image_label)
        layout.addWidget(name_label)
        widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed) # Fixed vertical size

        widget.setFixedWidth(size + 5) # Reduced space around image

        return widget

    def _format_damage_string(self, dph, dps, damage_type, dph_percent, direction: str) -> tuple[str, str]:
        """Formats DPH (with percent) and DPS strings."""
        percent_formatted = f"{dph_percent:.1f}%" # Format percentage first

        # Format DPH and add Percentage
        if dph == "不破防":
            dph_percent_str = f"DPH: <font color='red'>{dph}</font> (0.0%)" # Add 0% percent
        else:
            try:
               dph_val = float(dph)
               # New format: DPH: value (Percent: value%) (type)
               dph_percent_str = f"DPH: {dph_val:.2f} ({percent_formatted}) ({damage_type})"
            except (ValueError, TypeError):
               dph_percent_str = f"DPH: 无效 ({percent_formatted}) ({damage_type})" # Include percent even if DPH is invalid

        # Format DPS only
        try:
            dps_val = float(dps)
            dps_str = f"DPS: {dps_val:.2f}"
        except (ValueError, TypeError):
            dps_str = "DPS: 无效"

        # No need to combine based on direction anymore for DPS string
        # The direction is handled by the widget's alignment

        return dph_percent_str, dps_str # Return DPH+Percent and DPS separately


    def _populate_damage_rows(self):
        """Calculates and displays bidirectional damage info in rows."""
        if not self.other_monsters:
            self.rows_layout.addWidget(QLabel(f"{self.other_side_name} 没有目标怪物。"))
            return

        focused_monster = self.clicked_monster # Alias for clarity

        for target_monster in self.other_monsters:
            # --- Calculate Damage ---
            # Focused -> Target (Calculate all values)
            dph_f_to_t, dps_f_to_t, type_f_to_t, percent_f_to_t = calculate_damage(focused_monster, target_monster)
            # Format strings (Percent is now with DPH)
            dph_percent_f_to_t_str, dps_f_to_t_str = self._format_damage_string(
                dph_f_to_t, dps_f_to_t, type_f_to_t, percent_f_to_t, 'right' # Direction might still be useful for tooltips or future logic
            )
            tooltip_f_to_t = f"{focused_monster.name} 对 {target_monster.name}"

            # Target -> Focused (Calculate all values)
            dph_t_to_f, dps_t_to_f, type_t_to_f, percent_t_to_f = calculate_damage(target_monster, focused_monster)
            # Format strings (Percent is now with DPH)
            dph_percent_t_to_f_str, dps_t_to_f_str = self._format_damage_string(
                dph_t_to_f, dps_t_to_f, type_t_to_f, percent_t_to_f, 'left'
            )
            tooltip_t_to_f = f"{target_monster.name} 对 {focused_monster.name}"


            # --- Create Row Layout ---
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(3, 3, 3, 3) # Reduced padding within the row
            row_layout.setSpacing(3) # Further reduced spacing
            # Add border to the row widget itself for separation
            row_widget.setStyleSheet("QWidget { border: 1px solid #e0e0e0; border-radius: 4px; }")

            # --- Create Widgets for the Row ---
            focused_display = self._create_monster_display(focused_monster, size=65) # Reduced size
            target_display = self._create_monster_display(target_monster, size=65) # Reduced size

            # Arrow: Target -> Focused (Points Left <-) - Pass DPH+Percent and DPS separately
            arrow_t_to_f = DamageArrowWidget(dph_percent_t_to_f_str, dps_t_to_f_str, 'left', tooltip_t_to_f)
            # Arrow: Focused -> Target (Points Right ->) - Pass DPH+Percent and DPS separately
            arrow_f_to_t = DamageArrowWidget(dph_percent_f_to_t_str, dps_f_to_t_str, 'right', tooltip_f_to_t)

            # --- Add Widgets to Row Layout ---
            # Layout: [Focused Img] [Arrow <-] [Arrow ->] [Target Img]
            row_layout.addWidget(focused_display, stretch=0) # Fixed size
            row_layout.addWidget(arrow_t_to_f, stretch=1)    # Takes space
            row_layout.addWidget(arrow_f_to_t, stretch=1)    # Takes space
            row_layout.addWidget(target_display, stretch=0) # Fixed size

            # Add the complete row to the main vertical layout
            self.rows_layout.addWidget(row_widget)

        # Add a stretch at the end
        self.rows_layout.addStretch(1)

# Example usage (for testing purposes)
if __name__ == '__main__':
    import sys
    from PyQt6.QtWidgets import QApplication, QMessageBox # Import QMessageBox here

    # Create dummy Monster objects for testing
    clicked_monster_list = [
        Monster(id='src_m', name='测试源法术怪', template_name='obj_7', health=800, attack=200, defense=5, resistance=10, attack_interval=2.8, is_magic_damage=True)
    ]
    other_monsters_list = [
        Monster(id='tgt1', name='物理靶子', template_name='obj_2', health=500, attack=50, defense=20, resistance=10, attack_interval=2.0, is_magic_damage=False),
        Monster(id='tgt2', name='法术靶子', template_name='obj_3', health=500, attack=50, defense=5, resistance=30, attack_interval=2.0, is_magic_damage=True),
        Monster(id='tgt3', name='高防靶子', template_name='obj_4', health=1000, attack=10, defense=150, resistance=0, attack_interval=3.0, is_magic_damage=False),
    ]

    app = QApplication(sys.argv)

    # Simulate clicking on the 'left' monster (list length 1)
    window = DamageInfoWindow(clicked_monster_list, other_monsters_list, "左侧选中", "右侧全体")
    window.show()

    # # Example: Simulate clicking on a 'right' monster
    # left_all = [
    #     Monster(id='src_m', name='测试源法术怪', template_name='obj_7', health=800, attack=200, defense=5, resistance=10, attack_interval=2.8, is_magic_damage=True),
    #     Monster(id='src_p', name='测试源物理怪', template_name='obj_1', health=1000, attack=100, defense=10, resistance=0, attack_interval=1.5, is_magic_damage=False)
    # ]
    # right_clicked = [
    #      Monster(id='tgt1', name='物理靶子', template_name='obj_2', health=500, attack=50, defense=20, resistance=10, attack_interval=2.0)
    # ]
    # window2 = DamageInfoWindow(left_all, right_clicked, "左侧全体", "右侧选中")
    # window2.show()

    sys.exit(app.exec())