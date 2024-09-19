from qtpy import QtWidgets
from qtpy.QtGui import QColor, QIcon
from typing import Tuple
from qtpy.QtCore import Qt, QSize
from gui.main_window import MainWindow
import argparse
from pathlib import Path
import sys
import yaml

def start_app(config_path):
    app = QtWidgets.QApplication(sys.argv)
    # app.setWindowIcon(QIcon(str(Path.cwd() / Path("gui/assets/icon.png"))))
    ex = MainWindow(config_path=config_path)
    ex.show()
    sys.exit(app.exec_())


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]...", description="Start the result explorer"
    )

    parser.add_argument(
        "-v", "--version", action="version", version=f"{parser.prog} version 1.0.0"
    )
    parser.add_argument("--config", help="yaml file containing the configuration")
    parser.add_argument(
        "--beamline",
        dest="beamline",
        help="importer for the beamline (MX or LIX), default is MX",
        default="MX",
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

    start_app(config_path)


if __name__ == "__main__":
    main()