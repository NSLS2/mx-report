import getpass
import logging
import os
import typing

from pydantic import root_model
from pymongo import collection
import requests
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt, Signal, QSortFilterProxyModel
from utils.models import CollectionData, Sample

if typing.TYPE_CHECKING:
    from gui.main_window import MainWindow


class TreeFilterProxyModel(QtCore.QSortFilterProxyModel):
    def filterAcceptsRow(self, source_row, source_parent):
        source_model = self.sourceModel()
        index = source_model.index(source_row, 0, source_parent)

        # Check if the current item matches the filter
        if self.filterRegExp().indexIn(source_model.data(index)) >= 0:
            return True

        # Check if any of the child rows match the filter
        if self.has_matching_descendant(index):
            return True

        # Check if any ancestors match the filter (parent and above)
        if self.has_matching_ancestor(source_parent):
            return True

        return False

    def has_matching_descendant(self, index):
        """Recursively check if any child of this index matches the filter."""
        source_model = self.sourceModel()
        row_count = source_model.rowCount(index)

        # Check each child
        for i in range(row_count):
            child_index = source_model.index(i, 0, index)
            if self.filterRegExp().indexIn(source_model.data(child_index)) >= 0:
                return True
            if self.has_matching_descendant(child_index):
                return True

        return False

    def has_matching_ancestor(self, source_parent):
        """Check if any ancestors match the filter."""
        if not source_parent.isValid():
            return False  # We're at the root, no more ancestors

        source_model = self.sourceModel()
        if self.filterRegExp().indexIn(source_model.data(source_parent)) >= 0:
            return True

        # Recursively check the parent's parent (ancestor)
        return self.has_matching_ancestor(source_model.parent(source_parent))


class SampleTreeWidget(QtWidgets.QTreeView):
    itemDataClicked = Signal(object)

    def __init__(self, parent: "MainWindow|None" = None):
        super().__init__(parent)
        self.sample_model = QtGui.QStandardItemModel()
        self.setModel(self.sample_model)
        self.setStyleSheet("QTreeView::item::hover{background-color: #999966;}")
        self.setMaximumWidth(300)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # self.proxy_model = QSortFilterProxyModel(self)
        self.proxy_model = TreeFilterProxyModel(self)
        self.proxy_model.setSourceModel(self.sample_model)
        self.proxy_model.setFilterCaseSensitivity(
            Qt.CaseInsensitive
        )  # Optional: Ignore case sensitivity
        self.proxy_model.setFilterKeyColumn(-1)  # Search all columns

        # Set the proxy model as the model for the tree view
        self.setModel(self.proxy_model)

        header = self.header()

        header.setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents
        )  # Adjusts columns based on content
        header.setStretchLastSection(False)  # Prevents last column from stretching

        self.clicked.connect(self.on_item_clicked)

    def on_item_clicked(self, index):
        source_index = self.proxy_model.mapToSource(index)
        item = self.sample_model.itemFromIndex(source_index)
        if item is not None:
            item_data = item.data(Qt.UserRole)
            self.itemDataClicked.emit(item_data)

    def populate_tree(self, data: CollectionData):
        root = self.sample_model.invisibleRootItem()
        for puck_index, (puck_name, sample_list) in enumerate(data.puck_data.items()):
            puck_item = QtGui.QStandardItem(f"{puck_index+1}. {puck_name}")
            root.appendRow(puck_item)

            for sample_index, sample_name in enumerate(sample_list):
                collection_data = data.sample_collections[sample_name]
                sample_item = QtGui.QStandardItem(f"{sample_index+1}. {sample_name}")
                sample_item.setData(
                    {"sample_name": sample_name, "item_type": "sample"}, Qt.UserRole
                )
                puck_item.appendRow(sample_item)
                for standard_index, (standard_uid, standard_collection) in enumerate(
                    collection_data.standard.items()
                ):
                    standard_item = QtGui.QStandardItem(f"{standard_index+1}. Standard")
                    sample_item.appendRow(standard_item)
                    standard_item.setData(
                        {
                            "sample_name": sample_name,
                            "item_type": "standard",
                            "item_uid": standard_uid,
                        },
                        Qt.UserRole,
                    )
                    if standard_uid not in collection_data.rasters:
                        continue
                    for raster_index, (raster_uid, raster_collection) in enumerate(
                        collection_data.rasters[standard_uid].items()
                    ):
                        raster_item = QtGui.QStandardItem(
                            f"{'Orthogonal' if raster_index==0 else 'Face On'} Raster"
                        )
                        standard_item.appendRow(raster_item)
                        raster_item.setData(
                            {
                                "sample_name": sample_name,
                                "item_type": "raster",
                                "item_uid": raster_uid,
                                "standard_uid": standard_uid,
                            },
                            Qt.UserRole,
                        )

        self.expandAll()

    def on_filter_text_changed(self, text):
        # Update the filter in the proxy model
        self.proxy_model.setFilterFixedString(text)
        # Expand all items after filtering
        self.expandAll()


class SampleTree(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tree = SampleTreeWidget()
        self.filter_input = QtWidgets.QLineEdit()
        self.filter_input.setPlaceholderText("Filter samples...")
        self.setMaximumWidth(300)
        self.filter_input.textChanged.connect(self.tree.on_filter_text_changed)
        layout = QtWidgets.QVBoxLayout()

        layout.addWidget(self.filter_input)
        layout.addWidget(self.tree)
        self.setLayout(layout)
