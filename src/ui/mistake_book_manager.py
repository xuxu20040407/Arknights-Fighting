import json
import os
import sqlite3
import time
from typing import Dict, List, Optional, Tuple  # Added Tuple import

import numpy as np
from PyQt6.QtCore import Qt, QUrl, QPoint
from PyQt6.QtGui import QPixmap, QImage, QAction  # Added QAction
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox,
    QTextBrowser, QTextEdit, QSpinBox, QFormLayout,
    QListWidget, QListWidgetItem, QWidget, QRadioButton, QButtonGroup, QMenu  # Added QMenu
)

from src.core.log import logger  # Import the logger

# Assuming models and constants are accessible relative to the main execution context
# Adjust paths if necessary based on how this module is imported/used
try:
    # This path adjustment might be needed if run independently, but usually not when imported
    # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from src.models.monster import Monster
except ImportError:
    print("Error importing Monster class. Ensure paths are correct.")
    # Fallback or re-raise depending on desired behavior
    class Monster: pass # Dummy class to prevent immediate crash

# Define paths relative to this file's potential execution context
# It's often better to pass these paths from the main application instance
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
TEMPLATE_DIR = os.path.join(DATA_DIR, 'image')
DATABASE_PATH = os.path.join(DATA_DIR, 'mistakes.db')


class MistakeBookManager:
    """Handles database operations for the mistake book."""

    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initializes the SQLite database and creates the table if it doesn't exist."""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            # Add outcome column
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS mistakes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    left_monsters_json TEXT NOT NULL,
                    right_monsters_json TEXT NOT NULL,
                    notes TEXT,
                    outcome TEXT, -- Added field for outcome (e.g., 'left_win', 'right_win', 'draw')
                    left_monster_keys TEXT, -- Store sorted keys for easier matching
                    right_monster_keys TEXT -- Store sorted keys for easier matching
                )
            ''')
            # Check and add the outcome column if it doesn't exist (for existing databases)
            try:
                cursor.execute("SELECT outcome FROM mistakes LIMIT 1")
            except sqlite3.OperationalError:
                print("Adding 'outcome' column to existing mistakes table...")
                cursor.execute("ALTER TABLE mistakes ADD COLUMN outcome TEXT")
                conn.commit()
                print("'outcome' column added.")

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_left_keys ON mistakes (left_monster_keys)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_right_keys ON mistakes (right_monster_keys)")
            conn.commit()
            conn.close()
            logger.info(f"数据库初始化成功: {self.db_path}")
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            # Consider raising an exception or handling it more gracefully
            raise RuntimeError(f"Failed to initialize database: {e}") from e

    def load_all_mistakes(self) -> List[Dict]:
        """Loads all mistake book entries from the SQLite database."""
        entries = []
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row # Access columns by name
            cursor = conn.cursor()
            # Select the new outcome column
            cursor.execute("SELECT id, timestamp, left_monsters_json, right_monsters_json, notes, outcome FROM mistakes ORDER BY timestamp DESC")
            rows = cursor.fetchall()
            conn.close()

            for row in rows:
                try:
                    left_monsters = json.loads(row['left_monsters_json'])
                    right_monsters = json.loads(row['right_monsters_json'])
                    entries.append({
                        "id": row['id'],
                        "timestamp": row['timestamp'],
                        "left": {"monsters": left_monsters},
                        "right": {"monsters": right_monsters},
                        "notes": row['notes'],
                        "outcome": row['outcome'] # Load outcome
                    })
                except json.JSONDecodeError as json_e:
                    logger.warning(f"警告：无法解析数据库中的JSON数据 (ID: {row['id']}) - {json_e}")
                except Exception as parse_e:
                    logger.warning(f"警告：处理数据库行时出错 (ID: {row['id']}) - {parse_e}")

            logger.debug(f"从数据库成功加载 {len(entries)} 条错题记录。")
            return entries
        except sqlite3.Error as db_e:
            logger.error(f"数据库加载错误: {db_e}")
            QMessageBox.critical(None, "数据库加载错误", f"从数据库加载错题记录时出错：{db_e}")
            return []
        except Exception as e:
            logger.error(f"数据库加载失败: {e}")
            QMessageBox.critical(None, "数据库加载失败", f"加载错题记录时发生未知错误：{e}")
            return []

    def insert_mistake(self, entry_data: Dict) -> int | None:
        """Inserts a new mistake entry into the database."""
        timestamp = entry_data.get('timestamp', time.strftime("%Y-%m-%d %H:%M:%S"))
        left_monsters = entry_data.get('left', {}).get('monsters', {})
        right_monsters = entry_data.get('right', {}).get('monsters', {})
        notes = entry_data.get('notes', '')
        outcome = entry_data.get('outcome', 'unknown') # Get outcome

        left_json = json.dumps(left_monsters, sort_keys=True)
        right_json = json.dumps(right_monsters, sort_keys=True)
        left_keys_str = ",".join(sorted(left_monsters.keys()))
        right_keys_str = ",".join(sorted(right_monsters.keys()))

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            # Insert outcome value
            cursor.execute('''
                 INSERT INTO mistakes (timestamp, left_monsters_json, right_monsters_json, notes, outcome, left_monster_keys, right_monster_keys)
                 VALUES (?, ?, ?, ?, ?, ?, ?)
             ''', (timestamp, left_json, right_json, notes, outcome, left_keys_str, right_keys_str))
            conn.commit()
            new_id = cursor.lastrowid
            conn.close()
            logger.info(f"新错题记录已插入数据库，ID: {new_id}")
            return new_id
        except sqlite3.Error as db_e:
            logger.error(f"数据库插入错误: {db_e}")
            QMessageBox.critical(None, "数据库插入错误", f"插入错题记录时出错：{db_e}")
            return None
        except Exception as e:
            logger.error(f"数据库插入失败: {e}")
            QMessageBox.critical(None, "数据库插入失败", f"插入错题记录时发生未知错误：{e}")
            return None

    def find_matching_mistakes(self, current_left_types: List[str], current_right_types: List[str]) -> List[int]:
        """Checks if the current recognition matches any mistake book entries in the DB."""
        matching_ids = []
        current_left_keys_str = ",".join(sorted(current_left_types))
        current_right_keys_str = ",".join(sorted(current_right_types))

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            # Query using the indexed key strings, checking both direct and swapped matches
            cursor.execute('''
                SELECT id FROM mistakes
                WHERE (left_monster_keys = ? AND right_monster_keys = ?)
                   OR (left_monster_keys = ? AND right_monster_keys = ?)
            ''', (current_left_keys_str, current_right_keys_str, # Direct match
                  current_right_keys_str, current_left_keys_str)) # Swapped match
            rows = cursor.fetchall()
            conn.close()
            matching_ids = [row[0] for row in rows]
            return matching_ids
        except sqlite3.Error as db_e:
            logger.error(f"数据库查询错误 (匹配): {db_e}")
            QMessageBox.warning(None, "数据库查询错误", f"检查错题记录匹配时出错：{db_e}")
            return []
        except Exception as e:
            logger.error(f"匹配检查错误: {e}")
            QMessageBox.warning(None, "匹配检查错误", f"检查错题记录匹配时发生未知错误：{e}")
            return []

    def update_mistake(self, entry_id: int, updated_data: Dict) -> bool:
        """Updates an existing mistake entry in the database."""
        timestamp = updated_data.get('timestamp', time.strftime("%Y-%m-%d %H:%M:%S")) # Keep original or update? Let's update for now.
        left_monsters = updated_data.get('left', {}).get('monsters', {})
        right_monsters = updated_data.get('right', {}).get('monsters', {})
        notes = updated_data.get('notes', '')
        outcome = updated_data.get('outcome', 'unknown')

        left_json = json.dumps(left_monsters, sort_keys=True)
        right_json = json.dumps(right_monsters, sort_keys=True)
        left_keys_str = ",".join(sorted(left_monsters.keys()))
        right_keys_str = ",".join(sorted(right_monsters.keys()))

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE mistakes
                SET timestamp = ?, left_monsters_json = ?, right_monsters_json = ?, notes = ?, outcome = ?, left_monster_keys = ?, right_monster_keys = ?
                WHERE id = ?
            ''', (timestamp, left_json, right_json, notes, outcome, left_keys_str, right_keys_str, entry_id))
            conn.commit()
            updated_rows = cursor.rowcount
            conn.close()
            if updated_rows > 0:
                logger.info(f"错题记录 (ID: {entry_id}) 已更新。")
                return True
            else:
                logger.error(f"警告：未找到要更新的错题记录 (ID: {entry_id})。")
                return False
        except sqlite3.Error as db_e:
            logger.error(f"数据库更新错误 (ID: {entry_id}): {db_e}")
            QMessageBox.critical(None, "数据库更新错误", f"更新错题记录 (ID: {entry_id}) 时出错：{db_e}")
            return False
        except Exception as e:
            logger.error(f"数据库更新失败 (ID: {entry_id}): {e}")
            QMessageBox.critical(None, "数据库更新失败", f"更新错题记录 (ID: {entry_id}) 时发生未知错误：{e}")
            return False

    def delete_mistake(self, entry_id: int) -> bool:
        """Deletes a mistake entry from the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM mistakes WHERE id = ?", (entry_id,))
            conn.commit()
            deleted_rows = cursor.rowcount
            conn.close()
            if deleted_rows > 0:
                logger.info(f"错题记录 (ID: {entry_id}) 已删除。")
                return True
            else:
                # This might happen if the item was deleted between selection and confirmation
                logger.warning(f"警告：未找到要删除的错题记录 (ID: {entry_id})。")
                return False
        except sqlite3.Error as db_e:
            logger.error(f"数据库删除错误 (ID: {entry_id}): {db_e}")
            QMessageBox.critical(None, "数据库删除错误", f"删除错题记录 (ID: {entry_id}) 时出错：{db_e}")
            return False
        except Exception as e:
            logger.error(f"数据库删除失败 (ID: {entry_id}): {e}")
            QMessageBox.critical(None, "数据库删除失败", f"删除错题记录 (ID: {entry_id}) 时发生未知错误：{e}")
            return False


# --- Mistake Book Dialogs ---

class MistakeBookEntryDialog(QDialog):
    """Dialog for adding/editing a mistake book entry, handling monster counts."""
    # Updated type hints to accept tuples of (Monster, Optional[int])
    def __init__(self, left_monsters_with_counts: List[Tuple[Monster, Optional[int]]],
                 right_monsters_with_counts: List[Tuple[Monster, Optional[int]]],
                 all_monster_data: Dict[str, Monster], screenshot: Optional[np.ndarray] = None,
                 existing_entry: Optional[Dict] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("添加/编辑错题记录")
        self.all_monster_data = all_monster_data
        self.entry_data = {} # Stores the final data on accept
        self.existing_entry = existing_entry # Store for pre-filling

        # Determine initial monster quantities and notes from existing entry if provided
        initial_left_quantities = {}
        initial_right_quantities = {}
        initial_notes = ""
        initial_outcome = 'draw' # Default outcome for new entries
        if self.existing_entry:
            initial_left_quantities = self.existing_entry.get('left', {}).get('monsters', {})
            initial_right_quantities = self.existing_entry.get('right', {}).get('monsters', {})
            initial_notes = self.existing_entry.get('notes', '')
            initial_outcome = self.existing_entry.get('outcome', 'draw')

        layout = QVBoxLayout(self)

        # Display Screenshot
        if screenshot is not None and screenshot.size > 0:
            try:
                height, width, channel = screenshot.shape
                bytes_per_line = 3 * width
                q_img = QImage(screenshot.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
                pixmap = QPixmap.fromImage(q_img)
                screenshot_label = QLabel("捕获区域截图:")
                screenshot_display = QLabel()
                max_width = 400
                if pixmap.width() > max_width:
                     pixmap = pixmap.scaledToWidth(max_width, Qt.TransformationMode.SmoothTransformation)
                screenshot_display.setPixmap(pixmap)
                screenshot_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
                screenshot_display.setStyleSheet("border: 1px solid gray; margin-bottom: 10px;")
                layout.addWidget(screenshot_label)
                layout.addWidget(screenshot_display)
            except Exception as e:
                logger.error(f"错误：无法显示截图 - {e}")
                layout.addWidget(QLabel("<i>无法显示截图</i>"))

        form_layout = QFormLayout()

        # --- Outcome Selection ---
        outcome_group = QWidget()
        outcome_layout = QHBoxLayout(outcome_group)
        outcome_layout.setContentsMargins(0,0,0,0)
        self.outcome_button_group = QButtonGroup(self) # Group for radio buttons
        rb_left_win = QRadioButton("左胜")
        rb_right_win = QRadioButton("右胜")
        rb_draw = QRadioButton("平局/未知")
        # Set checked based on initial_outcome
        if initial_outcome == 'left_win':
            rb_left_win.setChecked(True)
        elif initial_outcome == 'right_win':
            rb_right_win.setChecked(True)
        else: # Default to draw/unknown
            rb_draw.setChecked(True)
        self.outcome_button_group.addButton(rb_left_win, 0) # ID 0 for left_win
        self.outcome_button_group.addButton(rb_right_win, 1) # ID 1 for right_win
        self.outcome_button_group.addButton(rb_draw, 2)     # ID 2 for draw/unknown
        outcome_layout.addWidget(rb_left_win)
        outcome_layout.addWidget(rb_right_win)
        outcome_layout.addWidget(rb_draw)
        outcome_layout.addStretch()
        form_layout.addRow(QLabel("<b>结果:</b>"), outcome_group)

        # Left Side Monsters
        self.left_widgets = {}
        left_group = QWidget()
        left_layout = QVBoxLayout(left_group)
        left_layout.addWidget(QLabel("<b>左侧怪物:</b>"))
        if left_monsters_with_counts:
            # Iterate through tuples (Monster, count)
            for monster, count_from_ocr in left_monsters_with_counts:
                hbox = QHBoxLayout()
                icon_label = QLabel()
                pixmap = QPixmap(os.path.join(TEMPLATE_DIR, f"{monster.template_name}.png"))
                if not pixmap.isNull():
                    icon_label.setPixmap(pixmap.scaled(32, 32, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                name_label = QLabel(f"{monster.name} ({monster.template_name})")
                spin_box = QSpinBox()
                spin_box.setRange(1, 99)
                # Set initial value: Use DB value if editing, else use OCR count (default 1)
                if self.existing_entry:
                    initial_value = initial_left_quantities.get(monster.template_name, 1)
                else:
                    # Use count from OCR if available, otherwise default to 1
                    initial_value = count_from_ocr if count_from_ocr is not None and count_from_ocr > 0 else 1
                spin_box.setValue(initial_value)
                hbox.addWidget(icon_label)
                hbox.addWidget(name_label)
                hbox.addStretch()
                hbox.addWidget(QLabel("数量:"))
                hbox.addWidget(spin_box)
                left_layout.addLayout(hbox)
                self.left_widgets[monster.template_name] = spin_box
        else:
            left_layout.addWidget(QLabel("无"))
        form_layout.addRow(left_group)

        # Right Side Monsters
        self.right_widgets = {}
        right_group = QWidget()
        right_layout = QVBoxLayout(right_group)
        right_layout.addWidget(QLabel("<b>右侧怪物:</b>"))
        if right_monsters_with_counts:
            # Iterate through tuples (Monster, count)
            for monster, count_from_ocr in right_monsters_with_counts:
                hbox = QHBoxLayout()
                icon_label = QLabel()
                pixmap = QPixmap(os.path.join(TEMPLATE_DIR, f"{monster.template_name}.png"))
                if not pixmap.isNull():
                    icon_label.setPixmap(pixmap.scaled(32, 32, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                name_label = QLabel(f"{monster.name} ({monster.template_name})")
                spin_box = QSpinBox()
                spin_box.setRange(1, 99)
                 # Set initial value: Use DB value if editing, else use OCR count (default 1)
                if self.existing_entry:
                    initial_value = initial_right_quantities.get(monster.template_name, 1)
                else:
                    # Use count from OCR if available, otherwise default to 1
                    initial_value = count_from_ocr if count_from_ocr is not None and count_from_ocr > 0 else 1
                spin_box.setValue(initial_value)
                hbox.addWidget(icon_label)
                hbox.addWidget(name_label)
                hbox.addStretch()
                hbox.addWidget(QLabel("数量:"))
                hbox.addWidget(spin_box)
                right_layout.addLayout(hbox)
                self.right_widgets[monster.template_name] = spin_box
        else:
            right_layout.addWidget(QLabel("无"))
        form_layout.addRow(right_group)

        # Notes
        self.notes_edit = QTextEdit()
        self.notes_edit.setPlaceholderText("添加笔记...")
        self.notes_edit.setPlainText(initial_notes) # Pre-fill notes
        form_layout.addRow(QLabel("<b>笔记:</b>"), self.notes_edit)

        layout.addLayout(form_layout)

        # Buttons
        button_box = QHBoxLayout()
        save_button = QPushButton("保存")
        cancel_button = QPushButton("取消")
        save_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        button_box.addStretch()
        button_box.addWidget(save_button)
        button_box.addWidget(cancel_button)
        layout.addLayout(button_box)

        self.setMinimumWidth(500)

    def accept(self):
        left_data = {template: spinbox.value() for template, spinbox in self.left_widgets.items()}
        right_data = {template: spinbox.value() for template, spinbox in self.right_widgets.items()}
        notes = self.notes_edit.toPlainText()

        # Get outcome based on checked radio button ID
        checked_id = self.outcome_button_group.checkedId()
        outcome = 'unknown' # Default
        if checked_id == 0:
            outcome = 'left_win'
        elif checked_id == 1:
            outcome = 'right_win'
        elif checked_id == 2:
            outcome = 'draw'

        self.entry_data = {
            "left": {"monsters": left_data},
            "right": {"monsters": right_data},
            "notes": notes,
            "outcome": outcome # Add outcome to data
        }
        super().accept()

    def get_entry_data(self) -> Optional[Dict]:
        return self.entry_data if self.result() == QDialog.DialogCode.Accepted else None


class MistakeBookQueryDialog(QDialog):
    """Dialog for viewing and managing mistake book entries with context menu."""
    # Add manager reference and highlight_ids
    def __init__(self, entries: List[Dict], all_monster_data: Dict[str, Monster], manager: MistakeBookManager, highlight_ids: Optional[List[int]] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("查询错题记录")
        self.entries = entries # Now includes 'id' key from DB
        self.all_monster_data = all_monster_data
        self.manager = manager # Store manager reference
        self.highlight_ids = highlight_ids # Store the IDs to highlight/scroll to

        layout = QHBoxLayout(self)

        # List View with Icons and Context Menu
        self.list_widget = QListWidget()
        self.list_widget.currentItemChanged.connect(self._display_entry_details)
        self.list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu) # Enable context menu
        self.list_widget.customContextMenuRequested.connect(self._show_context_menu) # Connect signal
        # self.list_widget.setSpacing(2) # Optional spacing

        for entry in entries:
            entry_id = entry.get('id', -1)
            left_monsters_dict = entry.get('left', {}).get('monsters', {})
            right_monsters_dict = entry.get('right', {}).get('monsters', {})

            # Create list item (without text initially)
            list_item = QListWidgetItem() # Create item first
            list_item.setData(Qt.ItemDataRole.UserRole, entry_id) # Store DB entry ID
            self.list_widget.addItem(list_item) # Add item to list

            # Create the custom widget for this item
            item_widget = self._create_list_item_widget(list(left_monsters_dict.keys()), list(right_monsters_dict.keys()))

            # Set the custom widget for the list item
            # Important: Set size hint for the item based on the widget's hint
            list_item.setSizeHint(item_widget.sizeHint())
            self.list_widget.setItemWidget(list_item, item_widget)

        # Increase stretch factor for list widget to make it wider (e.g., 3:2 ratio)
        layout.addWidget(self.list_widget, 2) # Adjusted stretch factor

        # Details View
        self.details_widget = QWidget()
        details_layout = QVBoxLayout(self.details_widget)
        self.details_left_label = QLabel("<b>左侧怪物:</b>")
        self.details_left_display = QTextBrowser()
        self.details_left_display.setOpenExternalLinks(False)
        self.details_right_label = QLabel("<b>右侧怪物:</b>")
        self.details_right_display = QTextBrowser()
        self.details_right_display.setOpenExternalLinks(False)
        self.details_outcome_label = QLabel("<b>结果:</b>") # Label for outcome
        self.details_outcome_display = QLabel("-") # Display for outcome
        self.details_notes_label = QLabel("<b>笔记:</b>")
        self.details_notes_display = QTextBrowser()

        details_layout.addWidget(self.details_left_label)
        details_layout.addWidget(self.details_left_display)
        details_layout.addWidget(self.details_right_label)
        details_layout.addWidget(self.details_right_display)
        details_layout.addWidget(self.details_outcome_label) # Add outcome label
        details_layout.addWidget(self.details_outcome_display) # Add outcome display
        details_layout.addWidget(self.details_notes_label)
        details_layout.addWidget(self.details_notes_display, 1) # Give notes stretch
        layout.addWidget(self.details_widget, 2) # Keep details stretch factor

        # Buttons
        button_box_details = QHBoxLayout()
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.accept)
        button_box_details.addStretch()
        button_box_details.addWidget(close_button)
        details_layout.addLayout(button_box_details)

        # Initial display and scrolling/highlighting
        # QListWidget is imported at the top of the file. ScrollHint is part of QListWidget.

        if self.entries:
            first_item_to_select = None
            # Find the first item matching the highlight IDs
            if self.highlight_ids:
                highlight_set = set(self.highlight_ids) # Use set for faster lookup
                for i in range(self.list_widget.count()):
                    item = self.list_widget.item(i)
                    entry_id = item.data(Qt.ItemDataRole.UserRole)
                    if entry_id in highlight_set:
                        first_item_to_select = item
                        break # Found the first one

            if first_item_to_select:
                # Scroll to and select the found item
                self.list_widget.scrollToItem(first_item_to_select, QListWidget.ScrollHint.PositionAtCenter)
                self.list_widget.setCurrentItem(first_item_to_select) # This triggers _display_entry_details
                logger.info(f"已跳转到错题记录 ID: {first_item_to_select.data(Qt.ItemDataRole.UserRole)}")
            elif self.list_widget.count() > 0:
                 # If no highlight IDs or no match found, select the first item
                 self.list_widget.setCurrentRow(0)
            else:
                 # If list is empty after population
                 self._clear_details()
        else:
            # If entries list was initially empty
            self._clear_details()

        self.setMinimumSize(900, 600) # Increase minimum size further

    def _create_list_item_widget(self, left_templates: List[str], right_templates: List[str]) -> QWidget:
        """Creates a widget displaying monster icons for the list view."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(4) # Slightly increase spacing
        icon_size = 48 # Increased icon size for list view

        # Left side icons
        left_icon_layout = QHBoxLayout()
        left_icon_layout.setSpacing(1)
        if left_templates:
            for template in sorted(left_templates): # Sort for consistency
                icon_label = QLabel()
                icon_path = os.path.join(TEMPLATE_DIR, f"{template}.png")
                pixmap = QPixmap(icon_path)
                if not pixmap.isNull():
                    icon_label.setPixmap(pixmap.scaled(icon_size, icon_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                else:
                    icon_label.setText("?") # Placeholder if icon missing
                    icon_label.setFixedSize(icon_size, icon_size)
                    icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                icon_label.setToolTip(template) # Show template name on hover
                left_icon_layout.addWidget(icon_label)
        else:
            left_icon_layout.addWidget(QLabel("-"))
        layout.addLayout(left_icon_layout)
        layout.addStretch(1) # Push "VS" to center

        # VS Label
        vs_label = QLabel("<b>VS</b>")
        layout.addWidget(vs_label)
        layout.addStretch(1) # Push right icons

        # Right side icons
        right_icon_layout = QHBoxLayout()
        right_icon_layout.setSpacing(1)
        if right_templates:
            for template in sorted(right_templates): # Sort for consistency
                icon_label = QLabel()
                icon_path = os.path.join(TEMPLATE_DIR, f"{template}.png")
                pixmap = QPixmap(icon_path)
                if not pixmap.isNull():
                    icon_label.setPixmap(pixmap.scaled(icon_size, icon_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                else:
                    icon_label.setText("?")
                    icon_label.setFixedSize(icon_size, icon_size)
                    icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                icon_label.setToolTip(template)
                right_icon_layout.addWidget(icon_label)
        else:
             right_icon_layout.addWidget(QLabel("-"))
        layout.addLayout(right_icon_layout)

        widget.setLayout(layout)
        # Set a minimum width to encourage the list view to be wider
        widget.setMinimumWidth(100) # Adjust minimum width for larger icons
        return widget

    def _show_context_menu(self, pos: QPoint):
        """Shows the right-click context menu for a list item."""
        item = self.list_widget.itemAt(pos)
        if item:
            menu = QMenu()
            edit_action = QAction("编辑此记录", self)
            delete_action = QAction("删除此记录", self)

            # Pass item or its ID to the handlers
            edit_action.triggered.connect(lambda: self._edit_selected_entry(item))
            delete_action.triggered.connect(lambda: self._delete_selected_entry(item))

            menu.addAction(edit_action)
            menu.addAction(delete_action)
            menu.exec(self.list_widget.mapToGlobal(pos))

    def _edit_selected_entry(self, item: QListWidgetItem):
        """Handles the 'Edit' action from the context menu."""
        entry_id = item.data(Qt.ItemDataRole.UserRole)
        if entry_id is None: return

        # Find the entry data
        entry_data = next((e for e in self.entries if e.get('id') == entry_id), None)
        if not entry_data:
            QMessageBox.warning(self, "错误", f"找不到 ID 为 {entry_id} 的记录进行编辑。")
            return

        # Recreate monster lists with counts from the saved entry data
        left_monsters_dict = entry_data.get('left', {}).get('monsters', {})
        right_monsters_dict = entry_data.get('right', {}).get('monsters', {})

        left_monsters_with_counts = []
        for template, count in left_monsters_dict.items():
            if template in self.all_monster_data:
                left_monsters_with_counts.append((self.all_monster_data[template], count))

        right_monsters_with_counts = []
        for template, count in right_monsters_dict.items():
            if template in self.all_monster_data:
                right_monsters_with_counts.append((self.all_monster_data[template], count))

        # Open the entry dialog, passing the existing data including counts
        dialog = MistakeBookEntryDialog(left_monsters_with_counts, right_monsters_with_counts, self.all_monster_data,
                                        screenshot=None, # No screenshot when editing
                                        existing_entry=entry_data, parent=self)

        if dialog.exec():
            updated_data = dialog.get_entry_data()
            if updated_data:
                # Update in the database via manager
                success = self.manager.update_mistake(entry_id, updated_data)
                if success:
                    # Refresh the entire list for simplicity, or update the specific item
                    self._refresh_list_and_details() # Reload data and update UI
                    QMessageBox.information(self, "成功", "记录已更新。")
                # else: Error message shown by manager

    def _delete_selected_entry(self, item: QListWidgetItem):
        """Handles the 'Delete' action from the context menu."""
        entry_id = item.data(Qt.ItemDataRole.UserRole)
        if entry_id is None: return

        entry_data = next((e for e in self.entries if e.get('id') == entry_id), None)
        timestamp = entry_data.get('timestamp', '未知时间') if entry_data else '未知记录'

        reply = QMessageBox.question(self, "确认删除",
                                     f"确定要删除这条记录吗？\n({timestamp})",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            success = self.manager.delete_mistake(entry_id)
            if success:
                # Remove item from list widget and internal list
                row = self.list_widget.row(item)
                self.list_widget.takeItem(row)
                self.entries = [e for e in self.entries if e.get('id') != entry_id]
                # Clear details if no items left or select next/previous
                if self.list_widget.count() == 0:
                    self._clear_details()
                # No need to explicitly select next/prev, currentItemChanged signal handles it
            # else: Error message shown by manager

    def _refresh_list_and_details(self):
         """Reloads data from manager and refreshes the list and details view."""
         print("Refreshing mistake book list...")
         self.entries = self.manager.load_all_mistakes() # Reload data
         self.list_widget.clear() # Clear existing list items
         # Re-populate the list widget
         for entry in self.entries:
             entry_id = entry.get('id', -1)
             left_monsters_dict = entry.get('left', {}).get('monsters', {})
             right_monsters_dict = entry.get('right', {}).get('monsters', {})
             list_item = QListWidgetItem()
             list_item.setData(Qt.ItemDataRole.UserRole, entry_id)
             self.list_widget.addItem(list_item)
             item_widget = self._create_list_item_widget(list(left_monsters_dict.keys()), list(right_monsters_dict.keys()))
             list_item.setSizeHint(item_widget.sizeHint())
             self.list_widget.setItemWidget(list_item, item_widget)

         # Select the first item if list is not empty, otherwise clear details
         if self.entries:
             self.list_widget.setCurrentRow(0) # This will trigger _display_entry_details
         else:
             self._clear_details()


    def _clear_details(self):
        self.details_left_display.clear()
        self.details_right_display.clear()
        self.details_outcome_display.setText("-") # Clear outcome display
        self.details_notes_display.clear()

    def _format_monster_details(self, monsters_dict: Dict[str, int]) -> str:
        """Formats monster details (icon, name, quantity) as HTML for QTextBrowser with larger elements."""
        html = ""
        if not monsters_dict:
            return "<p><i>无</i></p>"
        sorted_template_names = sorted(monsters_dict.keys())

        icon_size = 40 # Increased icon size
        font_size = 14 # Increased font size (using points for better scaling)

        for template_name in sorted_template_names:
            quantity = monsters_dict[template_name]
            monster_info = self.all_monster_data.get(template_name)
            name = monster_info.name if monster_info else template_name
            icon_path_abs = os.path.abspath(os.path.join(TEMPLATE_DIR, f"{template_name}.png")).replace("\\", "/")
            icon_url = QUrl.fromLocalFile(icon_path_abs).toString()

            # Use larger icon size and font size, adjust line-height
            html += f"<div style='margin-bottom: 5px; line-height: {icon_size}px;'>" # Adjust line-height
            img_tag = f"<img src='{icon_url}' width='{icon_size}' height='{icon_size}' style='vertical-align: middle;' alt='{template_name}'>" if os.path.exists(icon_path_abs) else "[无图]"
            html += f"{img_tag} "
            # Use pt for font size which might scale better than px in QTextBrowser
            html += f"<span style='font-size: {font_size}pt;'>{name} (x{quantity})</span>"
            html += f"</div>"
        return html

    def _display_entry_details(self, current_item: Optional[QListWidgetItem], previous_item: Optional[QListWidgetItem]):
        if not current_item:
            self._clear_details()
            return
        entry_id = current_item.data(Qt.ItemDataRole.UserRole)
        if entry_id is None or not isinstance(entry_id, int):
             self._clear_details()
             logger.error(f"错误：无效的条目 ID '{entry_id}'。")
             return
        entry = next((e for e in self.entries if e.get('id') == entry_id), None)
        if entry is None:
            self._clear_details()
            logger.error(f"错误：在加载的条目中找不到 ID 为 {entry_id} 的记录。")
            return
        try:
            left_monsters = entry.get('left', {}).get('monsters', {})
            right_monsters = entry.get('right', {}).get('monsters', {})
            notes = entry.get('notes', '')
            outcome = entry.get('outcome', 'unknown') # Get outcome

            # Format outcome for display with color and size
            outcome_color = "gray" # Default color
            outcome_text = "未知"
            if outcome == 'left_win':
                outcome_text = '左胜'
                outcome_color = 'green'
            elif outcome == 'right_win':
                outcome_text = '右胜'
                outcome_color = 'red'
            elif outcome == 'draw':
                outcome_text = '平局/未知'
                outcome_color = 'darkorange' # Or another color for draw

            outcome_font_size = 14 # Match monster name font size

            self.details_left_display.setHtml(self._format_monster_details(left_monsters))
            self.details_right_display.setHtml(self._format_monster_details(right_monsters))
            # Use rich text for outcome display
            self.details_outcome_display.setText(f"<font color='{outcome_color}' style='font-size: {outcome_font_size}pt;'><b>{outcome_text}</b></font>")
            self.details_notes_display.setPlainText(notes if notes else "无笔记")
        except Exception as e:
            logger.error(f"错误：显示错题记录详情时出错 (ID {entry_id}): {e}")
            self._clear_details()
            QMessageBox.warning(self, "显示错误", f"无法显示所选错题记录的详情：\n{e}")