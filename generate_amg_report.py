import argparse
from pathlib import Path
import pickle
import json
from typing import Optional
import utils
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
import multiprocessing
import time
import os

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
        default=True
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        help="Optionally specify the directory where you want to write the output files",
        default="amg_report"
    )
    return parser

def generate_html(context: dict, output_dir: Path):
    script_path = Path(__file__).resolve()
    template_path = script_path.parent / Path('templates/report_template.html')
    env = Environment(loader=FileSystemLoader(str(template_path.parent)))
    template = env.get_template('report_template.html')
    if context['current_page'] > 0:
        output_file_path = output_dir / Path(f"report_{context['current_page']}.html")
        for sample_name, td in context['toc_data'].items():
            if sample_name == "FastDP Summary Table":
                td['href'] = '../' + td['href'].split("/")[-1]
            else:
                td['href'] = "" + td['href'].split("/")[-1]

    else:
        output_file_path = output_dir / Path("report.html")
    html_content = template.render(context)

    #with open(f"./output/output_{context['current_page']}.html", 'w') as file:
    with output_file_path.open('w') as file:
        file.write(html_content)

def generate_toc_data(samples):
    toc_data = {"FastDP Summary Table": {"href": "report.html#summary_table", "sample_name": "FastDP Summary Table"},
    "AutoProc Summary Table": {"href": "output/report_1.html#autoproc_summary_table", "sample_name": "AutoProc Summary Table"},
    }
    for i, sample in enumerate(samples):
        sample_page = i//SAMPLES_PER_PAGE + 2
        sample_index_in_page = i%SAMPLES_PER_PAGE
        href = f"output/report_{sample_page}.html#sample-{sample}"
        toc_data[sample] =  {"href": href, "sample_name": sample}
    return toc_data

def generate_report_pages(full_data, samples, report_directory, report_output_directory, toc_data):
    
    total_pages = len(samples)//SAMPLES_PER_PAGE
    if len(samples)%SAMPLES_PER_PAGE:
        total_pages += 1

    beamline = next(iter(next(iter(full_data.values()))["standard"].values()))["request_obj"]["beamline"].upper()

    proposal = next(iter(full_data.values()))["sample"]["proposalID"]
    subtitle = f"Proposal: {proposal}  Beamline: {beamline}"

    context = {
            "title": "FastDP Report",
            "subtitle": subtitle,
            "toc_data": toc_data,
            "fastdp_column_names": ["Sample Path", "Hi", "Lo", "R_mrg", "cc12", "comp", "mult", "Hi", 
                                    "Lo", "R_mrg", "cc12", "comp", "mult", "symm", "a", "b", "c", 
                                    "alpha", "beta", "gamma"],
            "current_page": 0,
            "summary_table": True,
            "auto_proc_table": False,
            "full_data": full_data,
            "total_pages": None
    }
    generate_html(context, output_dir=report_directory)

    context = {
            "title": "AutoProc Report",
            "subtitle": subtitle,
            "toc_data": toc_data,
            "fastdp_column_names": ["Sample Path", "Hi", "Lo", "R_mrg", "cc12", "comp", "mult", "Hi", 
                                    "Lo", "R_mrg", "cc12", "comp", "mult", "symm", "a", "b", "c", 
                                    "alpha", "beta", "gamma"],
            "current_page": 1,
            "summary_table": False,
            "auto_proc_table": True,
            "full_data": full_data,
            "total_pages": None
    }
    generate_html(context, output_dir=report_output_directory)


    for page_num in range(total_pages):
        current_page = page_num + 2
        start_index = page_num*SAMPLES_PER_PAGE
        end_index = (current_page)*SAMPLES_PER_PAGE if len(samples) > (current_page)*SAMPLES_PER_PAGE else len(samples)
        print(start_index, end_index)
        context.update({
            "title": f"Report page #{current_page}",
            "subtitle": subtitle,
            "full_data": {sample: full_data[sample] for sample in samples[start_index: end_index]},
            "current_page": current_page,
            "summary_table": False,
            "auto_proc_table": False,
            "total_pages": total_pages
        })
        generate_html(context, output_dir=report_output_directory)


def process_single_sample(full_data, sample, collection_data_path, toc_data):
    sample_data = full_data[sample]
    for standard_id, standard_collection in sample_data['standard'].items():
        fast_dp_row = utils.get_standard_fastdp_summary(standard_collection['request_obj']['directory'])
        if fast_dp_row is None:
            diffraction_images = None
            fast_dp_row = (sample,) + ("-",) * 19
        else:
            diffraction_images = utils.create_grid_matplotlib_image(utils.get_standard_images(standard_collection['request_obj']['file_prefix'], 
                                                    standard_collection['request_obj']['directory']))
        fast_dp_row = list(fast_dp_row)
        fast_dp_row[0] = f'<a href="{toc_data.get(sample, {"href": "#"})["href"]}">{fast_dp_row[0]}</a>'
        auto_proc_row = utils.get_standard_autoproc_summary(standard_collection['request_obj']['directory'])
        auto_proc_row = auto_proc_row if auto_proc_row else (sample,) + ("-",) * 19
        auto_proc_row = list(auto_proc_row)
        auto_proc_row[0] = f'<a href="{toc_data.get(sample, {"href": "#"})["href"].split("/")[-1]}">{auto_proc_row[0]}</a>'
        #standard_collection.update({"diffraction_images": diffraction_images,
        #                            "fast_dp_row": fast_dp_row,
        #                            "auto_proc_row": auto_proc_row})
        standard_collection["diffraction_images"] = diffraction_images
        standard_collection["fast_dp_row"] = fast_dp_row
        standard_collection["auto_proc_row"] = auto_proc_row
        import pymongo
        client = pymongo.MongoClient(os.environ["MONGODB"])
        standard_result = client.analysisstore.analysis_header.find_one({"request": standard_collection['uid']})
        if standard_result:
            standard_collection['time'] = standard_result['time']
        standard_collection['time'] = utils.convert_epoch_to_datetime(standard_collection['time'])

        for raster_id, raster_req in sample_data['rasters'][standard_id].items():
            jpeg_path = utils.get_jpeg_path(raster_req, collection_data_path)
            raster_heatmap_data = utils.get_raster_spot_count(raster_req)
            i, j = None, None
            if max_index:=raster_req["request_obj"]['max_raster']['index'] is not None:
                i,j = utils.calculate_matrix_index(max_index, *utils.determine_raster_shape(raster_req["request_obj"]["rasterDef"]))
            heatmap_image = utils.create_matplotlib_image(raster_heatmap_data, i, j)
            lsdc_image = utils.encode_image_to_base64(jpeg_path)
            try:
                raster_req['time'] = utils.convert_epoch_to_datetime(int(raster_req['time']))
            except Exception as e:
                pass

            raster_req.update(
                {   "plot_image": heatmap_image, "jpeg_image": lsdc_image,
                }
            )
            # raster_req.update({"spot_reso_table": utils.process_rasters(raster_req["request_obj"]["directory"])})
            reso_table = utils.process_rasters(raster_req["request_obj"]["directory"])
            top_frames = list(reso_table["frame"][:3].to_numpy())
            top_frames_image = utils.get_spot_positions(raster_req, top_frames, reso_table)
            if top_frames_image is not None:
                raster_req.update({
                    "top_3_spots" : top_frames_image,
                    "hist_mean_neighbor_dist" : utils.create_histogram_image(reso_table, ["mean_neighbor_dist"], 
                                                'Distance ($\AA$)', 'No. of raster grid cells',
                                                'All raster cells 2-nearest neighbor'),
                    "hist_resolutions": utils.create_histogram_image(reso_table, ["max_resolution", "median_resolution"],
                                                                    "Distance ($\AA$)", "No. of raster grid cells",
                                                                    "All raster cells med/max res."),
                    "hist_spot_count": utils.create_histogram_image(reso_table, ["spot_count"],
                                                                    'Number of spots', "No. of raster grid cells", "All raster cells spot count")
                
                })
            else:
                raster_req.update({
                    "top_3_spots" : None,
                    "hist_mean_neighbor_dist" : None,
                    "hist_resolutions": None,
                    "hist_spot_count": None
                
                })
    full_data[sample] = sample_data
    #with lock:
    #    completed.value += 1
    print(f"Finished processing: {sample}")


def generate_report(report_output_directory: Path, 
                    report_data_directory: Path, 
                    data_dictionary: Optional[dict], 
                    json_data: Optional[dict],
                    database_json_file: Path,
                    data_pickle_file: Path,
                    collection_data_path: Path,
                    report_directory: Path):
    full_data = data_dictionary
    if full_data is None:
        print("Full data not found")
        if json_data is None:
            print("json data not found")
            utils.save_collection_data_to_disk(database_json_file, collection_data_path)
            json_data = utils.load_collection_data_from_disk(database_json_file)
        # full_data = json_data
    

    samples = [k for k, v in json_data.items() if v['rasters'] and v['standard']]
    # samples = samples[:11]
    toc_data = generate_toc_data(samples)

    num_processes = 8
    

    start_time = time.time()
    if full_data is None and json_data is not None:
        full_data = json_data

        with multiprocessing.Manager() as manager:
            # Initialize the shared dictionary with the existing dictionary
            full_data = manager.dict(full_data)
            completed = multiprocessing.Value('i', 0)
            lock = multiprocessing.Lock()

            # Create a pool of worker processes
            with multiprocessing.Pool(processes=num_processes) as pool:
                #with tqdm(total=len(samples)) as pbar:
                pool.starmap(process_single_sample, 
                                    [(full_data, sample, collection_data_path, toc_data) for sample in samples],
                                    #callback=lambda _: pbar.update(completed)
                                    )

            # Convert shared dictionary to a regular dictionary
            full_data = dict(full_data)  # Convert to a standard Python dict
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
        pickle.dump(full_data, f)

    generate_report_pages(full_data, samples, report_directory, report_output_directory, toc_data)

    

def main():
    parser = init_argparse()
    args = parser.parse_args()
    current_directory = Path.cwd()
    
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
            with data_pickle_file.open('rb') as f:
                try:
                    data_dictionary = pickle.load(f)
                except Exception as e:
                    data_dictionary = None
        if database_json_file.exists():
            with database_json_file.open("r") as jf:
                json_data = json.load(jf)


    generate_report(output_directory,
                    data_directory,
                    data_dictionary,
                    json_data,
                    database_json_file,
                    data_pickle_file,
                    current_directory,
                    report_directory
                    )

if __name__=="__main__":
    main()
