# import ipywidgets as widgets
import time
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
from amostra.client import commands as amostra_client
from analysisstore.client.commands import AnalysisClient
from utils.models import (
    Sample,
    Request,
    AutomatedCollection,
    CollectionData,
    RasterResult,
    RequestType,
    CollectionType,
    ManualCollection,
    StandardRequestDefinition,
    SampleName,
    PuckName,
    Result,
)
from typing import Generator, Dict, List, Any, Optional, Tuple
from uuid import UUID
import matplotlib.colors as mcolors

matplotlib.use("Agg")


def create_matplotlib_image(
    data: np.ndarray,
    max_index_x: Optional[int] = None,
    max_index_y: Optional[int] = None,
) -> str:
    # create_snake_arraye a Matplotlib plot
    plt.figure(figsize=(6, 4))
    fig, ax = plt.subplots()
    plt.imshow(data, cmap="inferno", interpolation="nearest")
    plt.colorbar()

    # Save the image to a bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)

    # Encode the image in base64
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close("all")
    buffer.close()

    return image_base64


def create_histogram_image(
    df: pd.DataFrame,
    columns: List[str],
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    color: str = "red",
) -> str:
    values = [df[column].to_numpy() for column in columns]
    plt.figure(figsize=(6, 4))
    try:
        plt.hist(values, bins=20, label=columns, histtype="stepfilled", color=color)
    except ValueError as e:
        print(e)
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


def create_grid_matplotlib_image(data: list[np.ndarray]) -> str:
    import matplotlib.pyplot as plt
    import io
    import base64

    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    axs = axs.flatten()

    # Display each image in the grid
    for i in range(len(data)):
        axs[i].imshow(data[i], norm=mcolors.LogNorm())
        # axs[i].axis("off")  # Hide the axes
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        for spine in axs[i].spines.values():
            spine.set_edgecolor("black")  # Set the color of the border
            spine.set_linewidth(2)

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


def get_standard_master_file(file_prefix: str, directory: Path) -> Optional[Path]:
    h5_path = Path(directory)
    master_file = None
    if not h5_path.exists():
        return master_file
    for file_path in h5_path.iterdir():
        if file_path.stem.startswith(file_prefix) and file_path.stem.endswith("master"):
            master_file = file_path
            break
    return master_file


def get_standard_images(file_prefix: str, directory: Path) -> list[np.ndarray]:
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


def get_standard_fastdp_summary(directory: Path) -> Optional[Any]:
    fastdp_file_path = Path(directory) / Path("fastDPOutput/fast_dp.xml")
    if fastdp_file_path.exists():
        output = parse_fdp_xml(str(fastdp_file_path))
        return output
    return None


def get_standard_autoproc_summary(directory: Path) -> Optional[Any]:
    autoproc_file_path = Path(directory) / Path("autoProcOutput/autoPROC.xml")
    autoproc_file_path1 = Path(directory) / Path("autoProcOutput1/autoPROC.xml")
    try:
        if autoproc_file_path.exists():
            return parse_fdp_xml(str(autoproc_file_path))
        elif autoproc_file_path1.exists():
            return parse_fdp_xml(str(autoproc_file_path1))
    except Exception as e:
        print(f"Exception while parsing autoProcOutput in {directory}")
    return None


def encode_image_to_base64(image_path: Path) -> str:
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


# client = pymongo.MongoClient(os.environ["MONGODB"])
# sample_db = client.amostra
# sample_collection = sample_db.sample
# request_collection = sample_db.request
# container_collection = sample_db.
amostra_db_params = {"host": os.environ["MONGODB"], "port": "7770"}
request_collection = amostra_client.RequestReference(**amostra_db_params)
sample_collection = amostra_client.SampleReference(**amostra_db_params)
container_collection = amostra_client.ContainerReference(**amostra_db_params)
analysis_store_client = AnalysisClient({"host": os.environ["MONGODB"], "port": "7773"})


def get_auto_collections(base_path: Path) -> Tuple[List[Request], Dict[str, Any]]:
    path_to_remove = Path("/nsls2/data4")
    try:
        base_path = base_path.relative_to(path_to_remove)
    except Exception as e:
        print(e)
    print(f"Getting collections in path: {base_path}")
    auto_collections = request_collection.find(
        **{
            "request_obj.directory": {
                "$regex": f"{base_path}"
            },
            "request_obj.centeringOption": "AutoRaster",
        }
    )
    auto_collections = [
        Request(**auto_collection) for auto_collection in auto_collections
    ]
    completed_standard_collections = analysis_store_client.find_analysis_header(
        **{"request": {"$in": [str(collection.uid) for collection in auto_collections]}}
    )
    completed_standard_collections = {
        collection_res["request"]: collection_res
        for collection_res in completed_standard_collections
    }
    return auto_collections, completed_standard_collections


def get_manual_collections(base_path: Path) -> Tuple[List[Request], Dict[str, Any]]:
    path_to_remove = Path("/nsls2/data4")
    manual_collections = request_collection.find(
        **{
            "request_obj.directory": {
                "$regex": f"{base_path.relative_to(path_to_remove)}"
            },
            "request_obj.centeringOption": "Interactive",
        }
    )
    # Manual collections include rasters, standards and vectors
    manual_collections = [
        Request(**manual_collection) for manual_collection in manual_collections
    ]
    # Get the collection results so that incomplete ones can be discarded
    completed_collections = analysis_store_client.find_analysis_header(
        **{
            "request": {
                "$in": [str(collection.uid) for collection in manual_collections]
            }
        }
    )
    collection_results = {
        collection_res["request"]: collection_res
        for collection_res in completed_collections
    }
    # Since these results also contain rasters we might as well parse it
    for collection in manual_collections:
        if str(collection.uid) in collection_results:
            if collection.request_type == RequestType.raster:
                collection.result = RasterResult(
                    **collection_results[str(collection.uid)]
                )
    return manual_collections, collection_results


def get_sample_data(sample_ids: set[str]) -> dict[SampleName, Sample]:
    sample_data: dict[SampleName, Sample] = {}
    sample_results = sample_collection.find(
        as_document=False, uid={"$in": list(sample_ids)}
    )
    for sample_result in sample_results:
        sample_result.pop("_id", None)
        sample = Sample(**sample_result)
        sample_data[sample.name] = sample
    return sample_data


def get_puck_data(
    sample_data: dict[SampleName, Sample],
) -> dict[PuckName, List[SampleName]]:
    puck_data = {}
    puck_ids = set()
    puck_id_data = defaultdict(list)
    for sample_name, sample in sample_data.items():
        puck_id_data[str(sample.container)].append(sample.name)
    print(f"Number of pucks: {len(puck_id_data)} from {len(sample_data)} samples")

    puck_results = container_collection.find(
        as_document=False, uid={"$in": list(puck_id_data.keys())}
    )
    for puck_result in puck_results:
        puck_data[puck_result["name"]] = reversed(
            list(puck_id_data[puck_result["uid"]])
        )

    print(f"Number of pucks found in DB: {len(puck_data)}")
    # Reversed because original order is reversed
    puck_data = dict(reversed(list(puck_data.items())))

    return puck_data


def process_manual_collections(
    manual_collections: List[Request], manual_collection_results: Dict[str, Any]
) -> CollectionData:
    """
    Processes the manual collection data received from amostra and validates
    it into a CollectionData pydantic model
    """
    all_data = {"sample_collections": {}, "puck_data": {}}
    sample_ids = set()
    puck_ids = set()
    puck_data: dict[PuckName, List[SampleName]] = defaultdict(list)
    requests: Dict[UUID, List[Request]] = defaultdict(list)
    for request in manual_collections:
        if request.request_def.directory.exists():
            requests[request.sample].append(request)
            sample_ids.add(str(request.sample))

    sample_data = get_sample_data(sample_ids)
    puck_data = get_puck_data(sample_data)

    sample_collections: Dict[SampleName, CollectionType] = {}

    for sample in sample_data.values():
        rasters: dict[UUID, Request] = {}
        standards: dict[UUID, Request] = {}
        vectors: dict[UUID, Request] = {}
        for request in requests[sample.uid]:
            if request.request_type == RequestType.raster:
                rasters[request.uid] = request
            elif request.request_type == RequestType.standard:
                standards[request.uid] = request
            elif request.request_type == RequestType.vector:
                vectors[request.uid] = request

        manual_collection = ManualCollection(
            sample=sample, rasters=rasters, standards=standards, vectors=vectors
        )
        sample_collections[sample.name] = manual_collection

    # This block of code tries to get the proposal number and beamline name
    # Because its not straightforward
    beamline = proposal = None
    for sample, manual_collection in sample_collections.items():
        collection_considered = None
        if manual_collection.standards:
            collection_considered = manual_collection.standards
        elif manual_collection.rasters:
            collection_considered = manual_collection.rasters
        elif manual_collection.vectors:
            collections_considered = manual_collection.vectors
        if collection_considered:
            for request in collection_considered.values():
                beamline = request.request_def.beamline.upper()
                break
        proposal = manual_collection.sample.proposal_id
        break

    collection_data = CollectionData(
        sample_collections=sample_collections,
        puck_data=puck_data,
        beamline=beamline,
        proposal=proposal,
    )
    return collection_data


def get_autoraster_data(
    standard_ids: set[str],
) -> Tuple[Dict[UUID, Dict[UUID, Request]], List[str]]:
    """
    Returns raster requests and results given standard collection UUIDs (as strings)
    Also returns a list of raster request uids (also as strings)
    """
    raster_data = request_collection.find(
        as_document=False,
        **{"request_obj.parentReqID": {"$in": list(standard_ids)}},
    )
    raster_requests: Dict[UUID, Dict[UUID, Request]] = defaultdict(dict)
    raster_req_uids = set()
    for raster in raster_data:
        raster.pop("_id", None)
        raster_request = Request(**raster)
        raster_requests[raster_request.request_def.parent_request_id][
            raster_request.uid
        ] = raster_request
        raster_req_uids.add(str(raster_request.uid))

    raster_result_data = get_raster_results(list(raster_req_uids))
    for raster_req_set in raster_requests.values():
        for raster_request in raster_req_set.values():
            raster_request.result = raster_result_data.get(raster_request.uid, None)

    return raster_requests, list(raster_req_uids)


def process_automated_collections(
    auto_collections: List[Request], auto_collection_results: Dict[str, Any]
) -> CollectionData:
    print("getting sample data and rasters")
    # all_data = defaultdict(dict)
    all_data = {"sample_collections": {}}
    sample_ids = set()
    puck_ids = set()
    requests: Dict[UUID, List[Request]] = defaultdict(list)
    standard_ids: Dict[str, Request] = {}

    for standard_request in auto_collections:
        # First make a dictionary of all standard collection to filter out ones
        # were not run (meaning no rasters were collected)
        standard_ids[str(standard_request.uid)] = standard_request

    print("Getting sample, puck and raster results from database...")
    start_time = time.time()
    raster_data, raster_req_uids = get_autoraster_data(set(standard_ids.keys()))
    for standard_id in raster_data.keys():
        # standard_id in raster_data tells us that atleast 1 raster was collected
        standard_request = standard_ids[str(standard_id)]
        requests[standard_request.sample].append(standard_request)
        sample_ids.add(str(standard_request.sample))
    
    sample_data = get_sample_data(sample_ids)
    puck_data = get_puck_data(sample_data)
    elapsed_time = time.time() - start_time
    print(f"Finished fetching database data. Time elapsed: {elapsed_time} seconds")

    
    
    sample_collections: dict[SampleName, CollectionType] = {}
    for sample in sample_data.values():
        std_collections: Dict[UUID, Request] = {}
        raster_collections: Dict[UUID, Dict[UUID, Request]] = {}
        for standard_request in requests[sample.uid]:
            raster_collections[standard_request.uid] = raster_data[standard_request.uid]
            std_collections[standard_request.uid] = standard_request

        auto_collection = AutomatedCollection(
            sample=sample, rasters=raster_collections, standard=std_collections
        )
        sample_collections[sample.name] = auto_collection

    beamline = next(
        iter(next(iter(sample_collections.values())).standard.values())
    ).request_def.beamline.upper()

    proposal = next(iter(sample_collections.values())).sample.proposal_id
    collection_data = CollectionData(
        sample_collections=dict(sample_collections),
        puck_data=puck_data,
        beamline=beamline,
        proposal=proposal,
    )
    return collection_data


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


def get_raster_results(request_uids: List[str]) -> Dict[UUID, RasterResult]:
    raster_result_data = analysis_store_client.find_analysis_header(
        **{"request": {"$in": request_uids}, "result_type": "rasterResult"}
    )
    raster_results = {}
    for raster_data in raster_result_data:
        raster_result = RasterResult(**raster_data)
        raster_results[raster_result.request] = raster_result

    return raster_results


def get_raster_spot_count(req, include_files=False):
    """
    client = pymongo.MongoClient(os.environ["MONGODB"])
    res_obj = client.analysisstore.analysis_header.find_one(
        {"request": str(req.uid), "result_type": "rasterResult"}
    )
    if res_obj is None:
        return np.zeros((2, 2))

    raster_result = RasterResult(**res_obj)
    req.result = raster_result
    """
    raster_result = req.result
    # res_obj = res_obj["result_obj"]["rasterCellResults"]["resultObj"]

    try:
        cells = raster_result.data.cell_data_collection.cells
        spot_counts = []
        file_locations = []
        for cell in cells:
            spot_counts.append(cell.spot_count)
            if include_files:
                file_locations.append(cell.image)
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
    if req.request_def.max_raster and req.request_def.max_raster.index is not None:
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
            try:
                df = pd.read_csv(path, skiprows=3, delimiter="\s+", header=None).to_numpy()[
                    :, 1:4
                ]
                diff = df - beam_center
                distances = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
                radius = np.max(distances)
                median_radius = np.median(distances)
                circle = Circle((0, 0), radius, color="blue", fill=False, linestyle="--")
                circle2 = Circle(
                    (0, 0), median_radius, color="green", fill=False, linestyle="--"
                )
                axs[i].add_patch(circle)
                axs[i].add_patch(circle2)
                matches = reso_table[reso_table["frame"] == idx]["max_resolution"]
                median_matches = reso_table[reso_table["frame"] == idx]["median_resolution"]
                if len(matches) > 0:
                    max_resolution = matches.values[0]
                    median_resolution = median_matches.values[0]
                else:
                    max_resolution = 0
                    median_resolution = 0
                axs[i].text(
                    0,
                    radius + 5,
                    f"{max_resolution:.2f} Å",
                    color="red",
                    bbox=dict(
                        facecolor="white",
                        edgecolor="black",
                        boxstyle="round,pad=0.5",
                        alpha=1,
                    ),
                    zorder=10,
                )
                axs[i].text(
                    0,
                    median_radius + 5,
                    f"{median_resolution:.2f} Å",
                    color="red",
                    bbox=dict(
                        facecolor="white",
                        edgecolor="black",
                        boxstyle="round,pad=0.5",
                        alpha=1,
                    ),
                    zorder=10,
                )

                axs[i].set_aspect("equal", adjustable="box")

                axs[i].scatter(
                    diff[:, 0], diff[:, 1], c=np.log(df[:, 2]), cmap="inferno", marker="."
                )
                axs[i].set_title(f"Frame {idx}")
                # axs[i].axis('off')  # Hide the axes
            except Exception as e:
                print(f"Could not generate spot images for {path} : {e}")
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


def save_collection_data_to_disk(json_path, data_path, collection_type):
    # if not Path(json_path).exists():
    if collection_type == "automated":
        all_data = process_automated_collections(*get_auto_collections(data_path))
    elif collection_type == "manual":
        all_data = process_manual_collections(*get_manual_collections(data_path))
    print("Writing data to disk")
    with open(json_path, "w") as f:
        # json.dump(all_data.json(), f, indent=4)
        f.write(all_data.model_dump_json(indent=4, by_alias=True))


def load_collection_data_from_disk(json_path) -> CollectionData:
    with open(json_path, "r") as f:
        full_data = json.load(f)
        full_data = CollectionData.model_validate(full_data)
        for sample in full_data.sample_collections.values():
            pass
    return full_data


def convert_epoch_to_datetime(epoch_time: float) -> str:
    dt = datetime.fromtimestamp(epoch_time)
    return dt.strftime("%Y-%m-%d %H:%M:%S")
