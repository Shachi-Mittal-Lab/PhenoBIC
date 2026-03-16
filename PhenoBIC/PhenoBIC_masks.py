"""
PhenoBIC cell phenotype inference from multiplex images (patch-based).

Reads per-patch multiplex TIFFs and cell segmentation TIFFs from a base directory.
Computes per-channel normalization percentiles from within-cell intensities.
Uses a multiprocessing pool to create an array of cell bounding boxcrops per batch followed
by PhenoBIC inference for each cell crop.
"""

import os
import skimage.io as io
import pandas as pd
import multiprocessing as mp
import numpy as np
import tensorflow as tf
from PIL import Image
from skimage.measure import regionprops_table
import json
from scipy import ndimage


def _path_tif_or_tiff(directory, base_name):
    """Return path to base_name.tif or base_name.tiff, whichever exists."""
    p_tif = os.path.join(directory, base_name + '.tif')
    p_tiff = os.path.join(directory, base_name + '.tiff')
    if os.path.isfile(p_tif):
        return p_tif
    if os.path.isfile(p_tiff):
        return p_tiff
    return p_tif  # let io.imread raise FileNotFoundError


def region_percentile(data, labels, percentile):
    """Compute a given percentile of pixel values per label region."""
    result = ndimage.labeled_comprehension(
        data, labels, np.unique(labels[labels > 0]),
        lambda x: np.percentile(x, percentile),
        float, 0
    )
    return result


def PhenoBIC_normalize(base_dir, channels, out_dir_path=None, lower_percentile=10.0, upper_percentile=90.0):
    """
    Compute per-channel normalization min/max from lower/upper percentiles of
    within-cell intensities across all patches. Writes JSON files to the output file path.
    """
    print(f"[PhenoBIC] Computing normalization percentiles ...")
    
    # Output file paths
    out_dir_path = os.path.join(base_dir, 'PhenoBIC_output')
    os.makedirs(out_dir_path, exist_ok=True)
    save_path_min = os.path.join(base_dir, 'PhenoBIC_output', 'normalization_min_values.json')
    save_path_max = os.path.join(base_dir, 'PhenoBIC_output', 'normalization_max_values.json')
    
    # Initialize dictionaries to store normalization values
    max_dict = {key: [] for key in channels}
    min_dict = {key: [] for key in channels}

    # Iterate over all patches
    i_patch = 0
    for patch in os.listdir(os.path.join(base_dir, 'multiplex_images')):
        i_patch += 1

        print(f"[PhenoBIC] Computing normalization percentiles for patch {i_patch} / {len(os.listdir(os.path.join(base_dir, 'multiplex_images')))}")
    
        # Read the cell segmentation image (label mask)
        cellseg_dir = os.path.join(base_dir, 'cell_segmentations')
        cellseg_path = _path_tif_or_tiff(cellseg_dir, patch)
        cellseg_img = io.imread(cellseg_path)

        # Iterate over all channels
        for channel in channels:
            # Read the patch image
            channel_dir = os.path.join(base_dir, 'multiplex_images', patch)
            patch_path = _path_tif_or_tiff(channel_dir, channel)
            patch_img = io.imread(patch_path)

            # Compute the upper percentile clip for the channel
            try:
                percentiles_max = region_percentile(patch_img, cellseg_img, upper_percentile)
                max_val = np.max(percentiles_max)
            except:
                max_val = -np.inf
            # Compute the lower percentile clip for the channel
            try:
                percentiles_min = region_percentile(patch_img, cellseg_img, lower_percentile)
                min_val = np.min(percentiles_min)
            except:
                min_val = np.inf

            max_dict[channel].append(float(max_val))
            min_dict[channel].append(float(min_val))

    # Add the minimum and maximum values for each channel to the dictionaries
    min_dict = {k: min(v) for k, v in min_dict.items()}
    max_dict = {k: max(v) for k, v in max_dict.items()}

    # Save min dictionary to JSON file
    with open(save_path_min, "w") as f:
        json.dump(min_dict, f)
    print(f"[PhenoBIC] Wrote min normalization: {save_path_min}")

    # Save max dictionary to JSON file
    with open(save_path_max, "w") as f:
        json.dump(max_dict, f)
    print(f"[PhenoBIC] Wrote max normalization: {save_path_max}")

    return None


def init_worker(patch_img_norm_):
    """Set the global normalized patch image for worker processes (used by cell_ROI_channel_extractor)."""
    global patch_img_norm
    patch_img_norm = patch_img_norm_


def cell_ROI_channel_extractor(label,
                            bounds_topleft_x,
                            bounds_topleft_y,
                            bounds_width,
                            bounds_height,
                            x_buffer,
                            y_buffer
                            ):
    """Extract one cell ROI, with a buffered bounding box, from the global normalized patch image."""
    global patch_img_norm
    ROI = patch_img_norm[max(0, bounds_topleft_y-y_buffer):
                            min(bounds_topleft_y+bounds_height+y_buffer, patch_img_norm.shape[0]-1),
                            max(0, bounds_topleft_x-x_buffer):
                            min(bounds_topleft_x+bounds_width+x_buffer, patch_img_norm.shape[1]-1)]
    return ROI, int(label)


def PhenoBIC_predict(base_dir, channels, model_path, num_cells_batch=4000, buffer_ratio=0.1, out_dir_path=None):
    """
    Load normalization JSONs and PhenoBIC model; run phenotype prediction per channel
    on cell ROIs in batches (multiprocessing + Keras). Writes classified phenotypes to output file path.
    """
    
    print(f"[PhenoBIC] Loading model: {os.path.basename(model_path)}")
    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    # Output file path
    out_dir_path = os.path.join(base_dir, 'PhenoBIC_output')
    os.makedirs(out_dir_path, exist_ok=True)
    class_save_path = os.path.join(out_dir_path, 'PhenoBIC_cell_phenotype_classes.csv')

    # Load the normalization JSON files
    with open(os.path.join(base_dir, 'PhenoBIC_output', 'normalization_max_values.json'), "r") as f:
        max_dict = json.load(f)
    with open(os.path.join(base_dir, 'PhenoBIC_output', 'normalization_min_values.json'), "r") as f:
        min_dict = json.load(f)

    # Initialize dictionaries to store classified phenotypes
    class_dict = {key: [] for key in channels + ['Patch', 'Cell_label']}
    probs_dict = {key: [] for key in channels + ['Patch', 'Cell_label']}

    # Initialize DataFrame to store cell bounding box coordinates
    measurements_df = pd.DataFrame({'label': [], 'Bounds_x': [], 'Bounds_y': [], 'Bounds_width': [], 'Bounds_height': []})

    i_patch = 0
    # Iterate over all patches
    for patch in os.listdir(os.path.join(base_dir, 'multiplex_images')):
        i_patch += 1
        print(f"[PhenoBIC] Processing patch {i_patch} / {len(os.listdir(os.path.join(base_dir, 'multiplex_images')))}: {patch}")

        # Read the cell segmentation image (label mask)
        cellseg_dir = os.path.join(base_dir, 'cell_segmentations')
        cellseg_path = _path_tif_or_tiff(cellseg_dir, patch)
        cellseg_img = io.imread(cellseg_path)

        # Compute the bounding box coordinates and sizes for each cell
        props = regionprops_table(
            cellseg_img,
            properties=('label', 'bbox')  # (min_row, min_col, max_row, max_col)
        )
        # Convert to DataFrame
        measurements_bounds_df = pd.DataFrame(props)
        # Compute derived bounding box coordinates and sizes
        measurements_bounds_df['Bounds_x'] = measurements_bounds_df['bbox-1']  # min_col
        measurements_bounds_df['Bounds_y'] = measurements_bounds_df['bbox-0']  # min_row
        measurements_bounds_df['Bounds_width'] = (
            measurements_bounds_df['bbox-3'] - measurements_bounds_df['bbox-1']  # max_col - min_col
        )
        measurements_bounds_df['Bounds_height'] = (
            measurements_bounds_df['bbox-2'] - measurements_bounds_df['bbox-0']  # max_row - min_row
        )
        # Keep only the required columns
        measurements_bounds_df = measurements_bounds_df[[
            'label', 'Bounds_x', 'Bounds_y', 'Bounds_width', 'Bounds_height'
        ]]
        measurements_bounds_df = measurements_bounds_df.sort_values('label', ascending=True).reset_index(drop=True)
        measurements_df = pd.concat([measurements_df, measurements_bounds_df], ignore_index=True)

        class_dict['Patch'].extend([patch] * len(measurements_bounds_df['label']))
        class_dict['Cell_label'].extend(measurements_bounds_df['label'])
        probs_dict['Patch'].extend([patch] * len(measurements_bounds_df['label']))
        probs_dict['Cell_label'].extend(measurements_bounds_df['label'])

        # Compute the number of cells and the number of iterations needed to process all cells
        num_cells = len(np.unique(cellseg_img)) - 1
        num_iters = (num_cells // num_cells_batch) + (num_cells % num_cells_batch != 0)
        i_channel = 0
        # Iterate over all channels
        for channel in channels:
            i_channel += 1
            print(f"[PhenoBIC] Processing channel {i_channel} / {len(channels)}: {channel}")
            # Read the patch image for the channel
            channel_dir = os.path.join(base_dir, 'multiplex_images', patch)
            patch_path = _path_tif_or_tiff(channel_dir, channel)
            patch_img = io.imread(patch_path)

            # Get the normalization clips for the channel
            min_int = min_dict[channel]
            max_int = max_dict[channel]

            # Linear normalization with clipping
            patch_img_norm = np.stack((np.uint8(np.clip((patch_img - min_int) / (max_int - min_int), 0, 1) * 255),) * 3, axis=-1)

            # Initialize lists to store classes and probabilities
            predictions_list = []
            probabilities_list = []

            # Iterate over the cells in batches
            for i in range(num_iters):

                start_idx = i * num_cells_batch
                end_idx = min(start_idx + num_cells_batch, num_cells)
                # Create a list of tuples for running cell ROI extraction with multiprocessing
                arg_tuples = list(zip(
                    measurements_bounds_df['label'].iloc[start_idx:end_idx],
                    (np.floor(measurements_bounds_df['Bounds_x'].iloc[start_idx:end_idx])).astype(int),
                    (np.floor(measurements_bounds_df['Bounds_y'].iloc[start_idx:end_idx])).astype(int),
                    (np.ceil(measurements_bounds_df['Bounds_width'].iloc[start_idx:end_idx])).astype(int),
                    (np.ceil(measurements_bounds_df['Bounds_height'].iloc[start_idx:end_idx])).astype(int),
                    (np.ceil(measurements_bounds_df['Bounds_width'].iloc[start_idx:end_idx] * buffer_ratio)).astype(int),
                    (np.ceil(measurements_bounds_df['Bounds_height'].iloc[start_idx:end_idx] * buffer_ratio)).astype(int)
                    ))
                # Create a pool of workers to extract the ROIs from the cells (multiprocessing)
                with mp.Pool(initializer=init_worker, initargs=(patch_img_norm,)) as pool:
                    results = pool.starmap(cell_ROI_channel_extractor, arg_tuples)
                
                # Unzip the results into two lists
                preprocessed_ROIs, labels = zip(*results)
                preprocessed_ROIs = list(preprocessed_ROIs)
                labels = list(labels)

                data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
                # Resize the ROIs to 48x48 pixels
                preprocessed_ROIs = np.array([Image.fromarray(img).resize((48,48), Image.NEAREST) for img in preprocessed_ROIs])

                # PhenoBIC inference
                predict_generator = data_gen.flow(
                    x=preprocessed_ROIs,
                    batch_size=32,
                    shuffle=False
                )
                pred = model.predict(predict_generator)
                pred = pred.reshape(-1)
                predictions= np.empty(len(pred), dtype='<U4')
                # Convert the predictions to binary cell expression labels
                predictions[pred <= 0.5] = 'neg'
                predictions[pred > 0.5] = 'pos'
                predictions_list.extend(predictions)
                print(f"[PhenoBIC] Processed {len(predictions_list)} / {num_cells} cells")
                probabilities_list.extend(pred)
            
            # Add the predictions to the class dictionary
            class_dict[channel].extend(predictions_list)
            print(f"[PhenoBIC] Processed all cells for patch: {patch} & channel: {channel}")

    class_df = pd.DataFrame(class_dict)

    class_df.to_csv(class_save_path, index_label=False)

    return None


def run_PhenoBIC(
        base_dir,
        channels,
        model_path,
        num_cells_batch=4000,
        buffer_ratio=0.1,
        lower_percentile=10.0,
        upper_percentile=90.0,
        out_dir_path=None):
    """
    Run full PhenoBIC pipeline for patch-based multiplex data: normalization then prediction.

    Expects base_dir to contain:
    - multiplex_images/<patch_name>/<channel>.tif (or .tiff)
    - cell_segmentations/<patch_name>.tif (or .tiff)

    First computes per-channel lower/upper percentile normalization across all patches and cells,
    then runs the PhenoBIC model on each channel for all cells and writes the output in a CSV file.

    Parameters
    ----------
    base_dir : str
        Path to base directory containing "multiplex_images/" and "cell_segmentations/" subdirectories.
    channels : list of str
        Channel names (e.g. ['CD3', 'CD8']) matching filenames <channel>.tif or <channel>.tiff under each patch folder.
    model_path : str
        Path to the PhenoBIC Keras model file (.keras).
    num_cells_batch : int, optional
        Number of cells per batch for inference (default 4000). Increase for speed at higher memory use.
    buffer_ratio : float, optional
        Buffer around each cell bounding box as fraction of box width/height (default 0.1 --> 10% buffering one each side).
    lower_percentile : float, optional
        Lower percentile clip for normalization (default 10.0, i.e. minimum of the 10th percentile within-cell channel intensity of all cells across all patches).
    upper_percentile : float, optional
        Upper percentile clip for normalization (default 90.0, i.e. maximum of the 90th percentile within-cell channel intensity of all cells across all patches).
    out_dir_path : str, optional
        Path to the output directory for normalization JSONs and results CSV; if None, uses base_dir/PhenoBIC_output.

    Outputs
    -------
    Writes to base_dir/PhenoBIC_output (or out_dir_path if set):
    - normalization_min_values.json, normalization_max_values.json: per-channel clip values.
    - PhenoBIC_cell_phenotype_classes.csv: Per-channel pos/neg predictions for each cell in each patch.
    """

    # Compute the normalization percentiles
    PhenoBIC_normalize(base_dir=base_dir,
                        channels=channels,
                        out_dir_path=None,
                        lower_percentile=lower_percentile,
                        upper_percentile=upper_percentile)
    # Run the PhenoBIC prediction
    PhenoBIC_predict(base_dir=base_dir,
                        channels=channels,
                        model_path=model_path,
                        num_cells_batch=num_cells_batch,
                        buffer_ratio=buffer_ratio,
                        out_dir_path=out_dir_path)
    
    return None
