import pathlib
import os
import h5py
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

raster_dir = ""


def gather_spot_files(raster_dir: pathlib.Path) -> list:
    """Gather a list of dozor spot files from a single raster scan."""
    spot_files = []
    for r, d, f in os.walk(raster_dir):
        for f_ in f:
            if f_.endswith("spot"):
                spot_files.append(f"{r}/{f_}")
    return spot_files


def extract_raster_metadata_h5(raster_dir: pathlib.Path) -> dict:
    """Get parameters for computing pixel reciprocal space dist and spot
    resolution from hdf5 file."""
    raster_metadata = {}
    for r, d, f in os.walk(raster_dir):
        for f_ in f:
            if "Raster" in f_ and "master.h5" in f_:
                h5 = h5py.File(f"{r}/{f_}")
                raster_metadata["x_pixel_size"] = h5[
                    "/entry/instrument/detector/x_pixel_size"
                ][()]
                raster_metadata["y_pixel_size"] = h5[
                    "/entry/instrument/detector/y_pixel_size"
                ][()]
                raster_metadata["detector_distance"] = h5[
                    "/entry/instrument/detector/detector_distance"
                ][()]
                raster_metadata["wavelength"] = h5[
                    "/entry/instrument/beam/incident_wavelength"
                ][()]
                raster_metadata["beam_center_x"] = h5[
                    "/entry/instrument/detector/beam_center_x"
                ][()]
                raster_metadata["beam_center_y"] = h5[
                    "/entry/instrument/detector/beam_center_y"
                ][()]

    if not raster_metadata:
        raise ValueError
    return raster_metadata


def get_points(spot_file):
    """dozor spot file to numpy array with columns: x,y, and intensity"""
    df1 = pd.read_table(spot_file, delimiter="\s+", header=None, skiprows=3)
    return df1.to_numpy()[:, 1:4]


def get_distances(points: np.array, experiment_metadata: dict) -> np.array:
    """Get k nearest neighbor distances for each spot. Will initially generate
    an Nxk array for N spots, which we then convert to reciprocal space units and
    filter to remove extreme outliers (very far or very close) and reshape to 1D
    for analysis. Use XDS method of sqrt(Qx*Qy)/(det_dist*wavelength) for per pixel dist."""
    points = points[:, :2]
    kdtree = KDTree(points)
    d, _ = kdtree.query(points, k=3)
    nonzero_distances = d[(d > 0)].reshape(-1)
    x_pixel_size = experiment_metadata["x_pixel_size"]
    y_pixel_size = experiment_metadata["y_pixel_size"]
    wl = experiment_metadata["wavelength"]
    det_dist = experiment_metadata["detector_distance"]
    distances = 1 / (
        nonzero_distances * np.sqrt(x_pixel_size * y_pixel_size) / (wl * det_dist)
    )
    filtered_distances = distances[(distances > 5) * (distances < 700)]
    return filtered_distances


def get_resolution(points: np.array, experiment_metadata: dict) -> np.array:
    """Convert x,y pairs from 2D detector into resolution."""
    points = points[:, :2]
    recentered_points = points - np.array(
        [experiment_metadata["beam_center_x"], experiment_metadata["beam_center_y"]]
    )
    r = np.sqrt(np.sum(recentered_points**2, -1))
    x_pixel_size = experiment_metadata["x_pixel_size"]
    y_pixel_size = experiment_metadata["y_pixel_size"]
    wl = experiment_metadata["wavelength"]
    det_dist = experiment_metadata["detector_distance"]
    #return 1 / (r * np.sqrt(x_pixel_size * y_pixel_size) / (wl * det_dist))
    return wl/(2*np.sin(0.5*np.arctan(r*np.sqrt(x_pixel_size*y_pixel_size)/det_dist)))


def process_rasters(raster_dir: pathlib.Path) -> pd.DataFrame:
    """Take a raster scan and convert to a dataframe with columns that contain
    resolution, spot count, and nearest neighbor distance. Once scan is processed
    then z-scores are computed for each feature and combined into a custom weighting function
    for the final score."""
    spot_files = gather_spot_files(raster_dir)
    experiment_metadata = extract_raster_metadata_h5(raster_dir)
    rows = []
    for sf in spot_files:
        frame = int(sf.split("/")[-1].split(".")[0])
        points = get_points(sf)
        res = get_resolution(points, experiment_metadata)
        distances = get_distances(points, experiment_metadata)
        raster_result = {
            "frame": frame,
            "median_resolution": np.median(res),
            "max_resolution": np.min(res),
            "mean_neighbor_dist": np.mean(distances),
            "spot_count": len(res),
        }
        rows.append(raster_result)
    df = pd.DataFrame(
        rows,
        columns=[
            "frame",
            "median_resolution",
            "max_resolution",
            "mean_neighbor_dist",
            "spot_count",
        ],
    )
    
    # compute custom z-score
    df['z_median_resolution'] = (np.mean(df['median_resolution']) - df['median_resolution']) / np.std(df['median_resolution'])
    df['z_mean_neighbor_dist'] = (np.mean(df['mean_neighbor_dist']) - df['mean_neighbor_dist']) / np.std(df['mean_neighbor_dist'])
    df['z_spot_count'] = (df['spot_count'] - np.mean(df['spot_count'])) / np.std(df['spot_count'])
    df['amx_score1'] = 0.6*df['z_median_resolution'] + 0.2*df['z_mean_neighbor_dist'] + 0.2*df['z_spot_count']

    df = df.sort_values(by='amx_score1', ascending=False)
    return df
