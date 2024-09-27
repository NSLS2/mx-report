from qtpy import QtWidgets
from qtpy.QtGui import QColor, QIcon
from typing import Tuple
from qtpy.QtCore import Qt, QSize
from gui.main_window import MainWindow
import argparse
from pathlib import Path
import sys
import yaml
import pickle

def start_app(config_path, collection_data, current_directory):
    app = QtWidgets.QApplication(sys.argv)
    # app.setWindowIcon(QIcon(str(Path.cwd() / Path("gui/assets/icon.png"))))
    ex = MainWindow(config_path=config_path, collection_data=collection_data, data_path = current_directory)
    ex.show()
    sys.exit(app.exec_())


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]...", description="Start the result explorer"
    )

    parser.add_argument(
        "-v", "--version", action="version", version=f"{parser.prog} version 0.0.1"
    )
    parser.add_argument("--config", help="yaml file containing the configuration")
    parser.add_argument(
        "--beamline",
        dest="beamline",
        help="importer for the beamline (MX or LIX), default is MX",
        default="MX",
    )
    parser.add_argument(
        "--report-dir",
        dest="report_dir",
        help="Optionally specify the directory where the report directory is",
        default="amg_report"
    )
    return parser


def main() -> None:
    parser = init_argparse()
    args = parser.parse_args()
    if not args.config:
        print("Please include the yaml file containing the app configuration")
        return
    config_path = Path(args.config)
    if not config_path.exists():
        print(
            f"Configuration file {config_path} does not exist, please provide a valid config path"
        )
        return
    current_directory = Path.cwd()
    report_directory = current_directory / Path(args.report_dir)
    data_directory = report_directory / Path("data")
    database_json_file = data_directory / Path("data.json")
    data_pickle_file = data_directory / Path("data.pickle")

    if not data_pickle_file.exists():
        print(f"Report data file not found at {data_pickle_file.absolute()}, please run the report script")
        return
    
    try:
        with data_pickle_file.open('rb') as f:
            collection_data = pickle.load(f)
    except Exception as e:
        print("Error importing data file: {e}.\nRecommend rerunning report script")

    start_app(config_path, collection_data, current_directory)


if __name__ == "__main__":
    main()