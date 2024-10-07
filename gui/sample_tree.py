import getpass
import logging
import os
import typing

import requests
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt, Signal
from utils.models import CollectionData, SampleData

if typing.TYPE_CHECKING:
    from gui.main_window import MainWindow


class SampleTree(QtWidgets.QTreeView):
    itemDataClicked = Signal(object)

    def __init__(self, parent: "MainWindow"):
        super().__init__(parent)
        self.sample_model = QtGui.QStandardItemModel()
        self.setModel(self.sample_model)
        self.setStyleSheet("QTreeView::item::hover{background-color: #999966;}")
        self.setMaximumWidth(300)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        header = self.header()

        header.setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents
        )  # Adjusts columns based on content
        header.setStretchLastSection(False)  # Prevents last column from stretching

        self.clicked.connect(self.on_item_clicked)

    def on_item_clicked(self, index):
        item = self.sample_model.itemFromIndex(index)
        item_data = item.data(Qt.UserRole)
        self.itemDataClicked.emit(item_data)

    def populate_tree(self, data: CollectionData):
        root = self.sample_model.invisibleRootItem()
        for index, (sample_name, collection_data) in enumerate(data.samples.items()):
            sample_item = QtGui.QStandardItem(f"{index+1}. {sample_name}")
            sample_item.setData(
                {"sample_name": sample_name, "item_type": "sample"}, Qt.UserRole
            )
            root.appendRow(sample_item)
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
                        f"{'Orthogonal' if raster_index==1 else 'Face On'} Raster"
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
