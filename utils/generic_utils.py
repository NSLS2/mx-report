# import ipywidgets as widgets
import pymongo
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import io
import base64
from PIL import Image
import cv2
import h5py
import hdf5plugin
from summarytable.summarytable import parse_fdp_xml
import pandas as pd
import json
from datetime import datetime
import matplotlib
import tqdm
import os
from utils.models import Sample, Request, SampleData, CollectionData, RasterResult

matplotlib.use("Agg")


# diff_img = f['/entry/data/data_000001'][50,:1700,1500:]
"""
def process_hdf_image(diff_img):
    # flatten saturated pixels (for better contrast enhancement)
    diff_img[diff_img == 65535] = 0

    # convert to 8bit gray scale for display
    diff_img_normalized = (diff_img - np.min(diff_img)) / (np.max(diff_img) - np.min(diff_img))
    diff_img_255 = (diff_img_normalized * 255).astype(np.uint8)

    # 2x downsample and contrast enhance
    diff_img_255_he = cv2.equalizeHist(cv2.pyrDown(cv2.pyrDown(diff_img_255)))
"""


def create_matplotlib_image(data, max_index_x=None, max_index_y=None):
    # create_snake_arraye a Matplotlib plot
    plt.figure(figsize=(6, 4))
    fig, ax = plt.subplots()
    plt.imshow(data, cmap="inferno", interpolation="nearest")
    """
    if max_index_x is not None and max_index_y is not None:
        rect = Rectangle((max_index_x - 0.5, max_index_y - 0.5), 1, 1, 
                         linewidth=2, edgecolor='green', facecolor='none',
                         label='Cell selected for standard collection')
        ax.add_patch(rect)
        plt.legend(handles=[rect], loc='upper right')
    """
    plt.colorbar()

    """
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            if data[row, col] > 0:
                ax.text(col,row,
                        calculate_flattened_index(row, col, data.shape[0], data.shape[1], "horizontal" if data.shape[1] > data.shape[0] else "vertical") + 1,
                        ha='center',
                        va="center",
                        color="white"
                        )
    """
    # Save the image to a bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)

    # Encode the image in base64
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close("all")
    buffer.close()
    return image_base64


def create_histogram_image(df, columns, xlabel="", ylabel="", title=""):
    values = [df[column].to_numpy() for column in columns]
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=20, label=columns)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if len(columns) > 1:
        # This assumes that we are plotting multiple histograms
        plt.legend()
        plt.minorticks_on()
        plt.axvline(x=1.5, color="red", linestyle=":", linewidth=2, label="x=1.5")
        plt.axvline(x=3, color="red", linestyle=":", linewidth=2, label="x=3")
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)

    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close("all")
    buffer.close()
    return image_base64


def create_grid_matplotlib_image(data):
    import matplotlib.pyplot as plt
    import io
    import base64

    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    axs = axs.flatten()

    # Display each image in the grid
    for i in range(len(data)):
        axs[i].imshow(data[i], cmap="gray")
        axs[i].axis("off")  # Hide the axes

    plt.tight_layout()
    # Save the image to a bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)

    # Encode the image in base64
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close("all")
    buffer.close()
    return image_base64


def get_standard_master_file(file_prefix, directory):
    h5_path = Path(directory)
    master_file = None
    if not h5_path.exists():
        return master_file
    for file_path in h5_path.iterdir():
        if file_path.stem.startswith(file_prefix) and file_path.stem.endswith("master"):
            master_file = file_path
            break
    return master_file


def get_standard_images(file_prefix, directory):
    standard_images = []
    master_file = get_standard_master_file(file_prefix, directory)
    if master_file is not None:
        # Open the HDF5 file
        with h5py.File(master_file, "r") as f:
            # Select 10 diff_imgs from the HDF5 file and process them

            number_of_files = len(f["/entry/data"])
            images_per_file = len(f["/entry/data/data_000001"])
            images_in_last_file = len(f[f"/entry/data/data_{number_of_files:06d}"])
            total_images = (number_of_files - 1) * images_per_file + images_in_last_file
            indices = np.linspace(0, total_images - 1, 10, dtype=int)
            for i in indices:
                # Select the image (adjust the slice index as needed)
                file_idx = i // images_per_file
                image_idx = i % images_per_file
                A = f[f"/entry/data/data_{file_idx + 1:06d}"][image_idx]
                m, n = A.shape
                diff_img = A[: m // 2, n // 2 :]

                # Flatten saturated pixels
                diff_img[diff_img == 65535] = 0

                # Convert to 8-bit grayscale
                diff_img_normalized = (diff_img - np.min(diff_img)) / (
                    np.max(diff_img) - np.min(diff_img)
                )
                diff_img_255 = (diff_img_normalized * 255).astype(np.uint8)

                # 2x downsample and contrast enhance
                diff_img_255_he = cv2.equalizeHist(
                    cv2.pyrDown(cv2.pyrDown(diff_img_255))
                )

                # Append the processed image to the list
                standard_images.append(diff_img_255_he)
            # print(f"Standard image len: {len(standard_images)}")
    return standard_images


def get_standard_fastdp_summary(directory):
    fastdp_file_path = Path(directory) / Path("fastDPOutput/fast_dp.xml")
    if fastdp_file_path.exists():
        return parse_fdp_xml(str(fastdp_file_path))
    return None


def get_standard_autoproc_summary(directory):
    autoproc_file_path = Path(directory) / Path("autoProcOutput/autoPROC.xml")
    autoproc_file_path1 = Path(directory) / Path("autoProcOutput1/autoPROC.xml")
    if autoproc_file_path.exists():
        return parse_fdp_xml(str(autoproc_file_path))
    elif autoproc_file_path1.exists():
        return parse_fdp_xml(str(autoproc_file_path1))
    return None


def encode_image_to_base64(image_path: Path):
    try:
        if image_path is None or not image_path.exists():
            raise ValueError("Invalid image path")

        with Image.open(image_path) as img:
            # Resize the image
            img = img.resize((300, 300))

            # Save the image to a bytes buffer
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            buffer.seek(0)

            # Encode the image in base64
            image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
            return image_base64

    except (AttributeError, ValueError, FileNotFoundError):
        # Create a blank image
        img = Image.new("RGB", (300, 300), color=(255, 255, 255))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)

        # Encode the blank image in base64
        image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        return image_base64


client = pymongo.MongoClient(os.environ["MONGODB"])
sample_db = client.amostra
sample_collection = sample_db.sample
request_collection = sample_db.request


def get_auto_collections(base_path):
    path_to_remove = Path("/nsls2/data4")
    regex_queries = [
        {"request_obj.directory": {"$regex": f"{path.relative_to(path_to_remove)}"}}
        for path in base_path.iterdir()
        if not path.stem.endswith("_dir")
    ]
    auto_collections = request_collection.find(
        {"$or": regex_queries, "request_obj.centeringOption": "AutoRaster"}
    )
    return auto_collections


def get_sample_data_and_rasters(auto_collections):
    print("getting sample data and rasters")
    # all_data = defaultdict(dict)
    all_data = {"samples": {}}
    for standard_collection in tqdm.tqdm(
        auto_collections,
        total=auto_collections.collection.count_documents(
            auto_collections._Cursor__spec
        ),
    ):
        sample = sample_collection.find({"uid": standard_collection["sample"]}).next()
        standard_collection.pop("_id", None)
        sample.pop("_id", None)

        sample = Sample(**sample)
        try:
            standard_collection = Request(**standard_collection)
        except Exception as e:
            print(standard_collection)
        rasters = list(
            request_collection.find(
                {"request_obj.parentReqID": str(standard_collection.uid)}
            )
        )
        all_data["samples"][sample.name] = {}
        all_data["samples"][sample.name]["sample"] = sample
        all_data["samples"][sample.name]["rasters"] = {
            standard_collection.uid: {}
        }  # defaultdict(dict)
        all_data["samples"][sample.name]["standard"] = {}
        for raster in rasters:
            raster.pop("_id", None)
            try:
                raster = Request(**raster)
            except Exception as e:
                print(e)
                print(raster)
            all_data["samples"][sample.name]["rasters"][standard_collection.uid][
                raster.uid
            ] = raster

        all_data["samples"][sample.name]["standard"][standard_collection.uid] = (
            standard_collection
        )

    all_data = CollectionData(**all_data)
    return all_data


def create_snake_array(flattened, M, N, raster_type):
    """
    Returns an MxN matrix of raster results given the
    flattened result array, direction, and shape of the raster
    """
    # Reshape the list to a 2D array
    if raster_type == "horizontal":
        # Reverse every even row for horizontal snaking
        array_2d = np.array(flattened).reshape(M, N)
        array_2d[1::2] = np.fliplr(array_2d[1::2])
    elif raster_type == "vertical":
        # Reverse every even column for vertical snaking
        array_2d = np.array(flattened).reshape(N, M)
        array_2d = array_2d.T
        array_2d[:, 1::2] = np.flipud(array_2d[:, 1::2])

    return array_2d


def determine_raster_shape(raster_def):
    """
    Returns the shape and direction of a raster given
    the raster definition
    """
    if (
        # raster_def["rowDefs"][0]["start"]["y"] == raster_def["rowDefs"][0]["end"]["y"]
        raster_def.row_defs[0].start.y == raster_def.row_defs[0].end.y
    ):  # this is a horizontal raster
        raster_dir = "horizontal"
    else:
        raster_dir = "vertical"

    num_rows = len(raster_def.row_defs)
    num_cols = raster_def.row_defs[0].num_steps
    if raster_dir == "vertical":
        num_rows, num_cols = num_cols, num_rows
    # print(raster_dir, num_rows, num_cols)
    return num_rows, num_cols, raster_dir


def get_raster_spot_count(req, include_files=False):
    client = pymongo.MongoClient(os.environ["MONGODB"])
    res_obj = client.analysisstore.analysis_header.find_one(
        {"request": str(req.uid), "result_type": "rasterResult"}
    )
    if res_obj is None:
        return np.zeros((2, 2))

    raster_result = RasterResult(**res_obj)
    req.result = raster_result
    # res_obj = res_obj["result_obj"]["rasterCellResults"]["resultObj"]
    cells = raster_result.data.cell_data_collection.cells
    spot_counts = []
    file_locations = []
    for cell in cells:
        spot_counts.append(cell.spot_count)
        if include_files:
            file_locations.append(cell.image)

    try:
        snake_array = create_snake_array(
            spot_counts, *determine_raster_shape(req.request_def.raster_def)
        )
    except Exception as e:
        print(f"Exception in creating snake array: {e}")
        print(f"Exception occured for request: {req}")
        snake_array = np.zeros((2, 2))
    if not include_files:
        return snake_array
    else:
        return snake_array, file_locations


def get_spot_positions(req: Request, indices, reso_table):
    # if req["request_obj"]["max_raster"]["index"] is not None:
    if req.request_def.max_raster.index is not None:
        # max_index = req["request_obj"]["max_raster"]["index"] + 1
        max_index = req.request_def.max_raster.index + 1
        indices = [max_index] + indices
    else:
        return None

    # beam_center = np.array([1555,1634,0])
    beam_center = np.array(
        [
            int(float(req.request_def.xbeam)),
            int(float(req.request_def.ybeam)),
            0,
        ]
    )

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs = axs.flatten()

    # Display each image in the grid
    for i, idx in enumerate(indices):
        row = (idx - 1) // req.request_def.raster_def.row_defs[0].num_steps
        path = Path(req.request_def.directory) / Path(f"dozor/row_{row}/{idx:05d}.spot")
        if path.exists():
            df = pd.read_csv(path, skiprows=3, delimiter="\s+", header=None).to_numpy()[
                :, 1:4
            ]
            diff = df - beam_center
            distances = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
            radius = np.max(distances)
            circle = Circle((0, 0), radius, color="red", fill=False, linestyle="--")
            circle2 = Circle(
                (0, 0), radius / 2, color="red", fill=False, linestyle="--"
            )
            axs[i].add_patch(circle)
            axs[i].add_patch(circle2)
            matches = reso_table[reso_table["frame"] == idx]["max_resolution"]
            if len(matches) > 0:
                max_resolution = matches.values[0]
            else:
                max_resolution = 0
            axs[i].text(0, radius + 5, f"Radius = {max_resolution:.2f}", color="blue")

            axs[i].set_aspect("equal", adjustable="box")

            axs[i].scatter(
                diff[:, 0], diff[:, 1], c=np.log(df[:, 2]), cmap="inferno", marker="."
            )
            axs[i].set_title(f"Frame {idx}")
            # axs[i].axis('off')  # Hide the axes
        else:
            print(f"File path {path} not found")

    plt.tight_layout()
    # Save the image to a bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)

    # Encode the image in base64
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close("all")
    buffer.close()
    return image_base64


def calculate_matrix_index(k, M, N, pattern="horizontal"):
    """
    Returns the row and column index in a raster based on the
    index of the result array, shape of the raster and the
    direction of collection
    """
    if pattern == "horizontal":
        i = k // N
        if i % 2 == 0:  # odd row
            j = k - (i * N)
        else:
            j = N - (k - (i * N)) - 1
        return i, j
    elif pattern == "vertical":
        j = k // M
        if j % 2 == 0:  # odd column
            i = k - (j * M)
        else:
            i = M - (k - (j * M)) - 1
        return i, j


def calculate_flattened_index(i, j, M, N, pattern="horizontal"):
    """
    Returns the index of the result array of a raster based on the
    row and column index, shape of the raster and the
    direction of collection
    """
    if pattern == "horizontal":
        if i % 2 == 0:  # odd row
            return i * (N) + j
        else:  # even row
            return i * N + (N - 1 - j)
    elif pattern == "vertical":
        if j % 2 == 0:  # Odd column
            print(i, j, M, N, j * M + i)
            return j * M + i
        else:  # Even column
            return j * M + (M - 1 - i)
    else:
        raise ValueError("Invalid pattern specified")


def get_jpeg_path(raster_req, base_path):
    """
    rel_data_dir = (
        Path(raster_req["request_obj"]["directory"])
        .resolve()
        .relative_to(Path(raster_req["request_obj"]["basePath"]).resolve())
    )
    """
    rel_data_dir = raster_req.request_def.directory.resolve().relative_to(
        raster_req.request_def.base_path.resolve()
    )
    new_full_path = base_path / Path("jpegs") / rel_data_dir
    new_full_path2 = base_path / Path("jpegs") / Path(*rel_data_dir.parts[1:])
    jpg_file = None
    for file in new_full_path.glob("*.jpg"):
        jpg_file = file
        break  # Exit loop after finding the first .jpg file
    if new_full_path2.exists():
        for file in new_full_path2.glob("*.jpg"):
            jpg_file = file
            break  # Exit loop after finding the first .jpg file
    return jpg_file


def save_collection_data_to_disk(json_path, data_path):
    # if not Path(json_path).exists():
    all_data = get_sample_data_and_rasters(get_auto_collections(data_path))
    with open(json_path, "w") as f:
        # json.dump(all_data.json(), f, indent=4)
        f.write(all_data.model_dump_json(indent=4, by_alias=True))


def load_collection_data_from_disk(json_path) -> CollectionData:
    with open(json_path, "r") as f:
        full_data = json.load(f)
        full_data = CollectionData.model_validate(full_data)
    return full_data


def convert_epoch_to_datetime(epoch_time: float) -> str:
    dt = datetime.fromtimestamp(epoch_time)
    return dt.strftime("%Y-%m-%d %H:%M:%S")
