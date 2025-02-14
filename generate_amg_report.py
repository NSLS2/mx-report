import argparse
from pathlib import Path
import pickle
import json
from typing import Optional, Dict, Any, List
import utils
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
import multiprocessing
import time
import os
from utils.models import (
    CollectionData,
    AutomatedCollection,
    CollectionType,
    ManualCollection,
    StandardResult,
    SampleName,
    PuckName,
    Request,
    RequestType,
)
import numpy as np
SAMPLES_PER_PAGE = 10


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] ...", description="Start the amg report generator"
    )
    parser.add_argument(
        "--regenerate",
        dest="regenerate",
        help="Regenerate the processed report data, useful when data was updated from the last time the report was generated",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        help="Optionally specify the directory where you want to write the output files",
        default="amg_report",
    )
    return parser


def generate_html(context: dict, output_dir: Path):
    script_path = Path(__file__).resolve()
    template_path = script_path.parent / Path("templates/report_template.html")
    env = Environment(loader=FileSystemLoader(str(template_path.parent)))
    template = env.get_template("report_template.html")
    if context["current_page"] > 0:
        output_file_path = output_dir / Path(f"report_{context['current_page']}.html")
        for sample_name, td in context["toc_data"].items():
            if sample_name == "FastDP Summary Table":
                td["href"] = "../" + td["href"].split("/")[-1]
            else:
                td["href"] = "" + td["href"].split("/")[-1]

    else:
        output_file_path = output_dir / Path("report.html")
    html_content = template.render(context)

    # with open(f"./output/output_{context['current_page']}.html", 'w') as file:
    with output_file_path.open("w") as file:
        file.write(html_content)


def generate_toc_data(puck_data: dict[str, list[str]]):
    toc_data = {
        "FastDP Summary Table": {
            "href": "report.html#summary_table",
            "sample_name": "FastDP Summary Table",
        },
        "AutoProc Summary Table": {
            "href": "output/report_1.html#autoproc_summary_table",
            "sample_name": "AutoProc Summary Table",
        },
    }
    # for i, sample in enumerate(samples):
    for i, (puck, samples) in enumerate(puck_data.items()):
        sample_page = i + 2
        for sample in samples:
            href = f"output/report_{sample_page}.html#sample-{sample}"
            toc_data[sample] = {"href": href, "sample_name": sample}
    return toc_data


def generate_report_pages(
    json_data, samples, report_directory, report_output_directory, toc_data
):
    full_data = json_data.sample_collections
    # total_pages = len(samples) // SAMPLES_PER_PAGE
    total_pages = len(json_data.puck_data)
    # if len(samples) % SAMPLES_PER_PAGE:
    #    total_pages += 1

    beamline = next(
        iter(next(iter(full_data.values())).standard.values())
    ).request_def.beamline.upper()

    proposal = next(iter(full_data.values())).sample.proposal_id
    subtitle = f"Proposal: {proposal}  Beamline: {beamline}"
    context = {
        "title": "FastDP Report",
        "subtitle": subtitle,
        "toc_data": toc_data,
        "fastdp_column_names": [
            "Sample Path",
            "Hi",
            "Lo",
            "R_mrg",
            "cc12",
            "comp",
            "mult",
            "Hi",
            "Lo",
            "R_mrg",
            "cc12",
            "comp",
            "mult",
            "symm",
            "a",
            "b",
            "c",
            "alpha",
            "beta",
            "gamma",
        ],
        "current_page": 0,
        "summary_table": True,
        "auto_proc_table": False,
        "full_data": full_data,
        "puck_data": json_data.puck_data,
        "total_pages": None,
    }
    generate_html(context, output_dir=report_directory)

    context = {
        "title": "AutoProc Report",
        "subtitle": subtitle,
        "toc_data": toc_data,
        "fastdp_column_names": [
            "Sample Path",
            "Hi",
            "Lo",
            "R_mrg",
            "cc12",
            "comp",
            "mult",
            "Hi",
            "Lo",
            "R_mrg",
            "cc12",
            "comp",
            "mult",
            "symm",
            "a",
            "b",
            "c",
            "alpha",
            "beta",
            "gamma",
        ],
        "current_page": 1,
        "summary_table": False,
        "auto_proc_table": True,
        "full_data": full_data,
        "puck_data": json_data.puck_data,
        "total_pages": None,
    }
    generate_html(context, output_dir=report_output_directory)

    for page_num in range(total_pages):
        current_page = page_num + 2
        # start_index = page_num * SAMPLES_PER_PAGE
        # end_index = (
        #    (current_page) * SAMPLES_PER_PAGE
        #    if len(samples) > (current_page) * SAMPLES_PER_PAGE
        #    else len(samples)
        # )
        # print(start_index, end_index)
        # print(full_data[samples[start_index]].result.)
        current_puck = list(json_data.puck_data.keys())[page_num]
        context.update(
            {
                "title": f"Report for {current_puck}",
                "subtitle": subtitle,
                "full_data": {
                    sample: full_data[sample]
                    for sample in json_data.puck_data[current_puck]
                },
                "current_page": current_page,
                "summary_table": False,
                "auto_proc_table": False,
                "total_pages": total_pages,
            }
        )
        generate_html(context, output_dir=report_output_directory)


def generate_standard_collection_report_data(
    standard_collection: Request, sample: str, toc_data
):
    fast_dp_row = utils.get_standard_fastdp_summary(
        standard_collection.request_def.directory
    )
    try:
        diffraction_images = None
        if fast_dp_row:
            diffraction_images = utils.create_grid_matplotlib_image(
                utils.get_standard_images(
                    standard_collection.request_def.file_prefix,
                    standard_collection.request_def.directory,
                )
            )
    except Exception as e:
        print(f"Exception in generating diffraction images: {e}")
        diffraction_images = None
    if fast_dp_row is None:
        # diffraction_images = None
        fast_dp_row = (sample,) + ("-",) * 19
    fast_dp_row = list(fast_dp_row)
    fast_dp_row[0] = (
        f'<a href="{toc_data.get(sample, {"href": "#"})["href"]}">{fast_dp_row[0]}</a>'
    )
    auto_proc_row = utils.get_standard_autoproc_summary(
        standard_collection.request_def.directory
    )
    auto_proc_row = auto_proc_row if auto_proc_row else (sample,) + ("-",) * 19
    auto_proc_row = list(auto_proc_row)
    auto_proc_row[0] = (
        f'<a href="{toc_data.get(sample, {"href": "#"})["href"].split("/")[-1]}">{auto_proc_row[0]}</a>'
    )
    result = StandardResult(
        diffraction_images=diffraction_images,
        fast_dp_row=fast_dp_row,
        auto_proc_row=auto_proc_row,
    )

    standard_collection.result = result
    return standard_collection

def generate_raster_report_data(raster_req: Request, collection_data_path:Path):
    
    jpeg_path = utils.get_jpeg_path(raster_req, collection_data_path)
    raster_heatmap_data = utils.get_raster_spot_count(raster_req)
    i, j = None, None
    if raster_req.request_def.max_raster:
        if max_index := raster_req.request_def.max_raster.index is not None:
            i, j = utils.calculate_matrix_index(
                max_index,
                *utils.determine_raster_shape(raster_req.request_def.raster_def),
            )
        else:
            i, j = None, None    
    heatmap_image = utils.create_matplotlib_image(raster_heatmap_data, i, j)
    lsdc_image = utils.encode_image_to_base64(jpeg_path)

    if raster_req.result:
        raster_req.result.plot_image = heatmap_image
        raster_req.result.jpeg_image = lsdc_image

    # raster_req.update({"spot_reso_table": utils.process_rasters(raster_req["request_obj"]["directory"])})
    reso_table = utils.process_rasters(raster_req.request_def.directory)
    top_frames = list(reso_table["frame"][:3].to_numpy())
    top_frames_image = utils.get_spot_positions(
        raster_req, top_frames, reso_table
    )
    if top_frames_image is not None and raster_req.result is not None:
        raster_req.result.top_frames = top_frames_image
        raster_req.result.hist_mean_neighbor_dist = (
            utils.create_histogram_image(
                reso_table,
                ["mean_neighbor_dist"],
                "Distance ($\AA$)",
                "No. of raster grid cells",
                "All raster cells 2-nearest neighbor",
            )
        )
        raster_req.result.hist_resolutions = utils.create_histogram_image(
            reso_table,
            ["max_resolution", "median_resolution"],
            "Distance ($\AA$)",
            "No. of raster grid cells",
            "All raster cells med/max res.",
            ["blue", "green"],
        )
        raster_req.result.hist_spot_count = utils.create_histogram_image(
            reso_table,
            ["spot_count"],
            "Number of spots",
            "No. of raster grid cells",
            "All raster cells spot count",
            "orange",
        )
    return raster_req

def process_single_collection(
    full_data: Dict[SampleName, CollectionType],
    #sample_data,
    sample: SampleName,
    collection_data_path,
    toc_data,
    lock,
):
    sample_data: CollectionType = full_data[sample]
    if isinstance(sample_data, AutomatedCollection):
        for standard_id, standard_collection in sample_data.standard.items():
            generate_standard_collection_report_data(standard_collection, sample, toc_data)
            for raster_id, raster_req in sample_data.rasters[standard_id].items():
                generate_raster_report_data(raster_req, collection_data_path)

    elif isinstance(sample_data, ManualCollection):
        for standard_collection in sample_data.standards.values():
            generate_standard_collection_report_data(standard_collection, sample, toc_data)
        for raster_req in sample_data.rasters.values():
            generate_raster_report_data(raster_req, collection_data_path)
    with lock:
        full_data[sample] = sample_data
    print(f"Finished processing: {sample}")


def generate_report(
    report_output_directory: Path,
    report_data_directory: Path,
    data_dictionary: Optional[dict],
    json_data: Optional[
        CollectionData | dict[str, AutomatedCollection] | dict[str, Any]
    ],
    database_json_file: Path,
    data_pickle_file: Path,
    collection_data_path: Path,
    report_directory: Path,
    collection_type="automated",
):
    full_data = data_dictionary
    if full_data is None:
        print("Full data not found")
        if json_data is None:
            print("json data not found")
            utils.save_collection_data_to_disk(
                database_json_file, collection_data_path, collection_type
            )
            json_data = utils.load_collection_data_from_disk(database_json_file)
        # full_data = json_data
    if isinstance(json_data, dict):
        json_data = CollectionData.model_validate(json_data)

    if full_data:
        json_data = full_data
        full_data = json_data.sample_collections

    # json_data = json_data.samples
    samples = [k for k, v in json_data.sample_collections.items()]
    # samples = samples[:11]
    toc_data = generate_toc_data(json_data.puck_data)

    num_processes = 8

    start_time = time.time()
    if full_data is None and json_data is not None:
        full_data = json_data.sample_collections

        with multiprocessing.Manager() as manager:
            # Initialize the shared dictionary with the existing dictionary
            full_data = manager.dict(full_data)
            completed = multiprocessing.Value("i", 0)
            queue = manager.Queue()
            lock = manager.Lock()

            # Create a pool of worker processes
            with multiprocessing.Pool(processes=num_processes) as pool:
                # with tqdm(total=len(samples)) as pbar:
                pool.starmap(
                    process_single_collection,
                    [
                        #(full_data[sample], sample, collection_data_path, toc_data, lock, queue)
                        (full_data, sample, collection_data_path, toc_data, lock)
                        for sample in samples
                    ],
                    # callback=lambda _: pbar.update(completed)
                )

            # Convert shared dictionary to a regular dictionary
            # full_data = dict(full_data)  # Convert to a standard Python dict
            while not queue.empty():
                full_data.update(queue.get())
            full_data = dict(full_data)
        json_data.sample_collections = full_data

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    # Format as HH:MM:SS
    print(f"Time taken: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
    """
    t = trange(len(samples), desc="Processing")
    for i in t:
        sample = samples[i]
        t.set_description(f"Processing sample {sample}")
        process_single_sample(full_data, sample, collection_data_path)
    """

    with data_pickle_file.open("wb") as f:
        pickle.dump(json_data, f)

    generate_report_pages(
        json_data, samples, report_directory, report_output_directory, toc_data
    )


def main():
    parser = init_argparse()
    args = parser.parse_args()
    current_directory = Path(os.environ.get("PWD", Path.cwd()))

    report_directory = current_directory / Path(args.output_dir)
    report_directory.mkdir(exist_ok=True)

    data_directory = report_directory / Path("data")
    data_directory.mkdir(exist_ok=True)

    output_directory = report_directory / Path("output")
    output_directory.mkdir(exist_ok=True)

    database_json_file = data_directory / Path("data.json")

    data_pickle_file = data_directory / Path("data.pickle")

    data_dictionary = None
    json_data = None

    if not args.regenerate:
        if data_pickle_file.exists():
            with data_pickle_file.open("rb") as f:
                try:
                    data_dictionary = pickle.load(f)
                except Exception as e:
                    data_dictionary = None
        if database_json_file.exists():
            with database_json_file.open("r") as jf:
                json_data = json.load(jf)

    generate_report(
        output_directory,
        data_directory,
        data_dictionary,
        json_data,
        database_json_file,
        data_pickle_file,
        current_directory,
        report_directory,
    )


if __name__ == "__main__":
    main()
