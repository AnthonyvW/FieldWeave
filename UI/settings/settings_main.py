from __future__ import annotations

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QDialog,
    QDialogButtonBox,
    QListWidget,
    QListWidgetItem,
    QStackedWidget,
    QFrame,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QScrollArea,
)

from .pages.camera_settings import camera_page
from .pages.automation_settings import automation_page
from .pages.machine_vision_settings import machine_vision_page
from .pages.navigation_settings import navigation_page
from .pages.about_settings import about_page

class SettingsButton(QToolButton):
    def __init__(self, tooltip: str = "Settings", parent: QWidget | None = None)-> None:
        super().__init__(parent)
        self.setToolTip(tooltip)
        self.setText("⚙")

        self.setAutoRaise(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedWidth(34)
        self.setFixedHeight(26)

class SettingsDialog(QDialog):
    # Signal emitted when user wants to save camera settings
    save_camera_settings = Signal()
    
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(860, 580)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        
        # Dark grey header bar spanning the top
        header = QFrame()
        header.setObjectName("SectionHeader")
        header.setFixedHeight(40)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 0, 16, 0)
        
        header_title = QLabel("Categories")
        header_title.setObjectName("SectionHeaderTitle")
        header_layout.addWidget(header_title)
        header_layout.addStretch()
        
        root.addWidget(header)
        
        # Main content area
        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Tree sidebar - flush with left and extends to bottom
        self.sidebar = QTreeWidget()
        self.sidebar.setFixedWidth(220)
        self.sidebar.setHeaderHidden(True)
        self.sidebar.setIndentation(15)

        # Pages container - flush with right edge and white background
        self.pages = QStackedWidget()
        self.pages.setStyleSheet("QStackedWidget { background: white; }")

        content_layout.addWidget(self.sidebar)
        content_layout.addWidget(self.pages)
        
        root.addWidget(content)

        # Bottom button bar with margins
        button_container = QWidget()
        button_container_layout = QHBoxLayout(button_container)
        button_container_layout.setContentsMargins(10, 10, 10, 10)
        
        button_box = QDialogButtonBox()
        
        self.save_btn = QPushButton("Save Settings")
        close_btn = QPushButton("Close")
        
        button_box.addButton(self.save_btn, QDialogButtonBox.ButtonRole.ActionRole)
        button_box.addButton(close_btn, QDialogButtonBox.ButtonRole.RejectRole)
        
        close_btn.clicked.connect(self._on_close_clicked)
        
        button_container_layout.addWidget(button_box)
        
        root.addWidget(button_container)
        
        # Store page widgets and their group boxes for scrolling
        self._page_widgets = {}
        self._group_boxes = {}  # Maps (page_name, group_name) -> QGroupBox widget

        self._add_page("Camera", camera_page(self))
        self._add_page("Navigation", navigation_page())
        self._add_page("Automation", automation_page())
        self._add_page("Machine Vision", machine_vision_page())
        self._add_page("About FieldWeave", about_page())

        self.sidebar.itemClicked.connect(self._on_tree_item_clicked)
        
        # Expand first item and select it
        if self.sidebar.topLevelItemCount() > 0:
            first_item = self.sidebar.topLevelItem(0)
            first_item.setExpanded(True)
            self.sidebar.setCurrentItem(first_item)
            self.pages.setCurrentIndex(0)
        
        # Initially disable save button
        self.save_btn.setEnabled(False)

    def open_to(self, category: str) -> None:
        for i in range(self.sidebar.topLevelItemCount()):
            item = self.sidebar.topLevelItem(i)
            if item and item.text(0) == category:
                self.sidebar.setCurrentItem(item)
                self._on_tree_item_clicked(item, 0)
                return
    
    def set_category_modified(self, category: str, modified: bool) -> None:
        """Update category text color to indicate modifications"""
        from PySide6.QtGui import QColor
        for i in range(self.sidebar.topLevelItemCount()):
            item = self.sidebar.topLevelItem(i)
            if item and item.text(0) == category:
                if modified:
                    # Use orange color (#f28c28 from style.py)
                    item.setForeground(0, QColor("#f28c28"))
                else:
                    # Reset to default
                    item.setData(0, Qt.ItemDataRole.ForegroundRole, None)
                return
    
    def _update_camera_groups(self, group_names: list[str]) -> None:
        """Update the Camera category tree item with actual group names"""
        # Find the Camera item
        for i in range(self.sidebar.topLevelItemCount()):
            item = self.sidebar.topLevelItem(i)
            if item and item.text(0) == "Camera":
                # Remove existing children
                item.takeChildren()
                
                # Add new children for each group
                page_index = item.data(0, Qt.ItemDataRole.UserRole)
                for group_name in group_names:
                    child_item = QTreeWidgetItem([group_name])
                    child_item.setData(0, Qt.ItemDataRole.UserRole, page_index)
                    child_item.setData(0, Qt.ItemDataRole.UserRole + 1, group_name)
                    item.addChild(child_item)
                
                # Re-expand the item if it was expanded
                if item.isExpanded() or self.sidebar.currentItem() == item:
                    item.setExpanded(True)
                
                return
    
    def register_group_box(self, page_name: str, group_name: str, group_box: QWidget) -> None:
        """Register a group box widget for a page so we can scroll to it"""
        self._group_boxes[(page_name, group_name)] = group_box
    
    @Slot(QTreeWidgetItem, int)
    def _on_tree_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """Handle tree item clicks"""
        # Get the stored data
        page_index = item.data(0, Qt.ItemDataRole.UserRole)
        group_name = item.data(0, Qt.ItemDataRole.UserRole + 1)
        
        if page_index is not None:
            # Switch to the page
            self.pages.setCurrentIndex(page_index)
            
            # If it's a group item, scroll to that group
            if group_name:
                # Get the page name from parent
                parent = item.parent()
                if parent:
                    page_name = parent.text(0)
                    group_box = self._group_boxes.get((page_name, group_name))
                    
                    if group_box:
                        # Find the scroll area in the current page
                        current_page = self.pages.currentWidget()
                        scroll_area = current_page.findChild(QScrollArea)
                        
                        if scroll_area:
                            # Scroll to the group box
                            scroll_area.ensureWidgetVisible(group_box)
    
    def _on_close_clicked(self) -> None:
        """Handle close button click with confirmation if settings modified"""
        # Check if camera page has unsaved changes
        camera_widget = self._page_widgets.get("Camera")
        if camera_widget and hasattr(camera_widget, 'has_unsaved_changes') and camera_widget.has_unsaved_changes():
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved camera settings. Do you want to save them?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
            )
            
            if reply == QMessageBox.StandardButton.Cancel:
                return  # Don't close
            elif reply == QMessageBox.StandardButton.Yes:
                # Save settings before closing
                self.save_camera_settings.emit()
        
        self.reject()

    def _add_page(self, name: str, page: QWidget) -> None:
        """Add a page and create tree items for it and its groups"""
        page_index = self.pages.addWidget(page)
        self._page_widgets[name] = page
        
        # Create parent tree item for the category
        parent_item = QTreeWidgetItem([name])
        parent_item.setData(0, Qt.ItemDataRole.UserRole, page_index)
        parent_item.setData(0, Qt.ItemDataRole.UserRole + 1, None)  # No group name for parent
        self.sidebar.addTopLevelItem(parent_item)
        
        # Find all group boxes in the page and add them as child items
        if hasattr(page, 'get_group_names'):
            # If the page provides a method to get group names
            groups = page.get_group_names()
            for group_name in groups:
                child_item = QTreeWidgetItem([group_name])
                child_item.setData(0, Qt.ItemDataRole.UserRole, page_index)
                child_item.setData(0, Qt.ItemDataRole.UserRole + 1, group_name)
                parent_item.addChild(child_item)