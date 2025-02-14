from qtpy import QtWidgets
from qtpy.QtGui import QColor, QIcon, QMouseEvent, QPixmap
from typing import Tuple
from qtpy.QtCore import Qt, QSize
import yaml
from gui.albula.interface import AlbulaInterface
import sys
import json
import utils
from pathlib import Path
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from gui.sample_tree import SampleTree
from gui.table_widgets import SummaryTable, CollectionTable
from utils.models import CollectionData, StandardRequestDefinition


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.updateGeometry()
        self.colorbar = None


class MainWindow(QtWidgets.QMainWindow):
    def __init__(
        self, *args, config_path, collection_data: CollectionData, data_path, **kwargs
    ):
        self.config_path = config_path
        self.full_data = collection_data
        self.data_path = data_path
        try:
            with self.config_path.open("r") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(
                f"Exception occured while reading config file {self.config_path}: {e}"
            )
            raise e
        self.config = config
        super().__init__(*args, **kwargs)
        self.setWindowTitle(f"Data explorer at {self.config.get('beamline', '99id1')}")
        self.albulaInterface = AlbulaInterface(
            python_path="/nsls2/software/mx/daq/conda_env/lsdc-37/bin/python"
        )
        # self.albulaInterface.open_file("/nsls2/data/amx/proposals/2024-1/pass-314921/314921-20240224-dtime/mx314921-1/tlys-676/1/FGZ-009_1/tlys-676_10289_master.h5")
        # self.data_path = Path("/nsls2/data/amx/proposals/2024-2/pass-312346/312346-20240801-ragusa_dartmouth_standby/mx312346-1")

        # json_path = "dartmouth.json"
        # self.full_data = self.load_collection_data_from_disk(json_path)
        """
        for standard_id, standard_collection in full_data[sample]['standard'].items():
            for raster_id, raster_req in full_data[sample]['rasters'][standard_id].items():
                jpeg_path = utils.get_jpeg_path(raster_req, fgz_data_path)
                raster_heatmap_data, self.image_files = utils.get_raster_spot_count(raster_req, include_files=True)
                self.raster_req = raster_req
        """

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        layout = QtWidgets.QGridLayout()
        central_widget.setLayout(layout)

        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        layout.addWidget(self.canvas, 0, 1, 1, 1)

        self.sample_tree = SampleTree(self)

        layout.addWidget(self.sample_tree, 0, 0, 3, 1)
        self.sample_tree.tree.populate_tree(self.full_data)
        self.sample_tree.tree.itemDataClicked.connect(self.handle_tree_clicked)

        self.sample_cam_image = QtWidgets.QLabel(self)
        layout.addWidget(self.sample_cam_image, 0, 2, 1, 1)

        self.summary_table = SummaryTable(processing_type_col=True)
        groupBox = QtWidgets.QGroupBox("Sample Summary Table")
        gb_layout = QtWidgets.QVBoxLayout()
        gb_layout.addWidget(self.summary_table)
        groupBox.setLayout(gb_layout)
        layout.addWidget(groupBox, 1, 1, 1, 2)

        self.table_tab_widget = QtWidgets.QTabWidget()

        self.collection_table = CollectionTable()
        self.full_summary_table = SummaryTable()
        self.autoproc_summary_table = SummaryTable()
        self.table_tab_widget.addTab(self.collection_table, "Collection Table")
        self.table_tab_widget.addTab(self.full_summary_table, "FastDP Summary Table")
        self.table_tab_widget.addTab(
            self.autoproc_summary_table, "AutoProc Summary Table"
        )

        self.populate_full_summary_table()

        layout.addWidget(self.table_tab_widget, 2, 1, 1, 2)
        self.data = None

        # Connect the click event to a custom handler
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_hover)

    def populate_full_summary_table(self):
        for puck_name, sample_names in self.full_data.puck_data.items():
            for sample_name in sample_names:
                data = self.full_data.sample_collections[sample_name]
                # for sample_name, data in self.full_data.samples.items():
                for standard_uid, standard_req in data.standard.items():
                    # fast_dp_row = utils.get_standard_fastdp_summary(standard_req['request_obj']['directory'])
                    # fast_dp_row = fast_dp_row if fast_dp_row is not None else (sample_name,) + ("N/A",) * 19
                    blank_row = [sample_name,] + ["-",] * 19
                    fast_dp_row = (
                        standard_req.result.fast_dp_row
                        if (standard_req.result and standard_req.result.fast_dp_row)
                        else blank_row
                    )
                    fast_dp_row[0] = sample_name
                    self.full_summary_table.add_data(fast_dp_row)
                    blank_row_autoproc = [sample_name,] + ["-",] * 19
                    auto_proc_row = (
                        standard_req.result.auto_proc_row
                        if (standard_req.result and standard_req.result.auto_proc_row)
                        else blank_row_autoproc
                    )
                    if not auto_proc_row:
                        auto_proc_row = blank_row_autoproc
                    auto_proc_row[0] = sample_name
                    self.autoproc_summary_table.add_data(auto_proc_row)

    def handle_tree_clicked(self, data: dict):
        if data["item_type"] == "raster":
            self.canvas.setHidden(False)
            self.sample_cam_image.setHidden(False)
            self.summary_table.setHidden(True)
            raster_req = self.full_data.sample_collections[data["sample_name"]].rasters[
                data["standard_uid"]
            ][data["item_uid"]]
            self.data, self.image_files = utils.get_raster_spot_count(
                raster_req, include_files=True
            )
            self.raster_req = raster_req
            self.plot_heatmap()

            image_path = utils.get_jpeg_path(raster_req, self.data_path)
            pixmap = QPixmap(str(image_path))

            self.sample_cam_image.setPixmap(
                pixmap.scaledToWidth(300, Qt.SmoothTransformation)
            )
            self.collection_table.clear_data()
            self.collection_table.add_data(
                [
                    raster_req.uid,
                    raster_req.request_time,
                    raster_req.request_def.sweep_start,
                    raster_req.request_def.sweep_end,
                    raster_req.request_def.img_width,
                    raster_req.request_def.exposure_time,
                    raster_req.request_def.detector_distance,
                ]
            )

        if data["item_type"] == "standard":
            self.canvas.setHidden(True)
            self.sample_cam_image.setHidden(True)
            self.summary_table.setHidden(False)
            standard_req = self.full_data.sample_collections[data["sample_name"]].standard[
                data["item_uid"]
            ]
            master_file_path = utils.get_standard_master_file(
                standard_req.request_def.file_prefix,
                standard_req.request_def.directory,
            )
            if master_file_path is not None:
                self.albulaInterface.open_file(str(master_file_path))
            self.summary_table.clear_data()
            # fast_dp_row = utils.get_standard_fastdp_summary(standard_req['request_obj']['directory'])
            # fast_dp_row = fast_dp_row if fast_dp_row is not None else (data['sample_name'],) + ("N/A",) * 19
            fast_dp_row = standard_req.result.fast_dp_row

            auto_proc_row = standard_req.result.auto_proc_row

            if not auto_proc_row:
                auto_proc_row = [
                    data["sample_name"],
                ] + [
                    "-",
                ] * 19
            if not fast_dp_row:
                fast_dp_row = [
                    data["sample_name"],
                ] + [
                    "-",
                ] * 19

            fast_dp_row[0] = data["sample_name"]
            auto_proc_row[0] = data["sample_name"]

            self.summary_table.add_data(
                [
                    "FastDP result",
                ]
                + fast_dp_row
            )
            self.summary_table.add_data(
                [
                    "AutoProc result",
                ]
                + auto_proc_row
            )
            self.collection_table.clear_data()
            self.collection_table.add_data(
                [
                    standard_req.uid,
                    standard_req.request_time,
                    standard_req.request_def.sweep_start,
                    standard_req.request_def.sweep_end,
                    standard_req.request_def.img_width,
                    standard_req.request_def.exposure_time,
                    standard_req.request_def.detector_distance,
                ]
            )

    def plot_heatmap(self):
        # Clear previous plots

        if self.canvas.colorbar:
            self.canvas.colorbar.remove()
        self.canvas.axes.clear()

        if max_index := self.raster_req.request_def.max_raster.index:
            i, j = utils.calculate_matrix_index(
                max_index,
                *utils.determine_raster_shape(self.raster_req.request_def.raster_def),
            )
            rect = Rectangle(
                (j - 0.5, i - 0.5),
                1,
                1,
                linewidth=2,
                edgecolor="green",
                facecolor="none",
            )
            self.canvas.axes.add_patch(rect)

        # Create the heatmap
        cax = self.canvas.axes.imshow(self.data, cmap="inferno", origin="upper")

        y_ticks = np.arange(self.data.shape[0])
        x_ticks = np.arange(self.data.shape[1])

        # Label offset: adjust ticks to be between points
        self.canvas.axes.set_xticks(x_ticks)
        self.canvas.axes.set_yticks(y_ticks)

        self.canvas.axes.set_xlim([x_ticks[0] - 0.5, x_ticks[-1] + 0.5])
        self.canvas.axes.set_ylim([y_ticks[-1] + 0.5, y_ticks[0] - 0.5])

        divider = make_axes_locatable(self.canvas.axes)
        cax_cb = divider.append_axes(
            "right", size="5%", pad=0.05
        )  # Adjust size and padding as needed
        self.canvas.colorbar = self.canvas.figure.colorbar(cax, cax=cax_cb)

        # Create a text annotation for the tooltip, initially hidden
        self.tooltip = self.canvas.axes.text(
            -2.5,
            0.95,
            "",
            color="white",
            backgroundcolor="black",
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(facecolor="black", alpha=0.8),
        )
        self.tooltip.set_visible(False)

        # Create a rectangle for highlighting cells, initially hidden
        self.highlight = Rectangle(
            (0, 0), 1, 1, linewidth=2, edgecolor="white", facecolor="none"
        )
        self.canvas.axes.add_patch(self.highlight)
        self.highlight.set_visible(False)

        # Redraw the canvas
        self.canvas.draw()
        self.canvas.figure.tight_layout()

    def on_hover(self, event):
        if self.data is not None:
            matrix = self.data
            # Check if the mouse is over the axes
            if event.inaxes == self.canvas.axes:
                # Get the row and column indices
                x, y = event.xdata, event.ydata
                col, row = int(np.floor(x)), int(np.floor(y))

                # Check if the indices are within the bounds of the matrix
                if 0 <= row < matrix.shape[0] and 0 <= col < matrix.shape[1]:
                    # Get the intensity of the current cell
                    intensity = matrix[row, col]

                    # Update the position and text of the tooltip
                    # self.tooltip.set_position((col+5, row+5))
                    self.tooltip.set_text(
                        f"({row}, {col})\nSpot Count: {intensity:.2f}"
                    )
                    self.tooltip.set_visible(True)

                    # Update the position of the highlight rectangle
                    self.highlight.set_bounds(col - 0.5, row - 0.5, 1, 1)
                    self.highlight.set_visible(True)
                else:
                    # Hide the tooltip and highlight if outside the matrix
                    self.tooltip.set_visible(False)
                    self.highlight.set_visible(False)
            else:
                # Hide the tooltip and highlight if the mouse is outside the axes
                self.tooltip.set_visible(False)
                self.highlight.set_visible(False)

            self.canvas.draw_idle()  # Redraw the canvas

    def on_click(self, event):
        # Check if the click is inside the axes
        if event.inaxes is not None:
            # Get the x and y pixel coordinates of the click
            x, y = int(round(event.xdata)), int(round(event.ydata))
            print(event.xdata, event.ydata)

            # Get the ValueErrorue at the clicked position
            if 0 <= x < self.data.shape[1] and 0 <= y < self.data.shape[0]:
                value = self.data[y, x]
                print(f"Clicked on: x={x}, y={y}, value={value}")
                idx = utils.calculate_flattened_index(
                    y,
                    x,
                    *utils.determine_raster_shape(
                        self.raster_req.request_def.raster_def
                    ),
                )

                print(idx)
                print(self.image_files[idx])
                self.albulaInterface.open_file(self.image_files[idx])
        else:
            print("Clicked outside axes bounds")

    def closeEvent(self, event):
        self.albulaInterface.close()
        event.accept()
        sys.exit()

    def load_collection_data_from_disk(self, json_path):
        with open(json_path, "r") as f:
            full_data = json.load(f)
        return full_data
