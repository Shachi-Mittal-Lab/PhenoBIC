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


def region_percentile(data, labels, percentile):
    result = ndimage.labeled_comprehension(
        data, labels, np.unique(labels[labels > 0]),
        lambda x: np.percentile(x, percentile),
        float, 0
    )
    return result


def PhenoBIC_normalize(base_dir, channels, out_dir_path=None):
    
    out_dir_path = os.path.join(base_dir, 'PhenoBIC_output')
    os.makedirs(out_dir_path, exist_ok=True)

    save_path_min = os.path.join(base_dir, 'PhenoBIC_output', 'normalization_min_values.json')
    save_path_max = os.path.join(base_dir, 'PhenoBIC_output', 'normalization_max_values.json')
    
    max_dict = {key: [] for key in channels}
    min_dict = {key: [] for key in channels}

    for patch in os.listdir(os.path.join(base_dir, 'multiplex_images')):
    
        cellseg_path = os.path.join(os.path.join(base_dir, 'cell_segmentations', patch + '.tiff'))
        cellseg_img = io.imread(cellseg_path)

        for channel in channels:

            print(patch, channel)

            patch_path = os.path.join(base_dir, 'multiplex_images', patch, channel + '.tif')
            patch_img = io.imread(patch_path)


            try:
                percentiles_max = region_percentile(patch_img, cellseg_img, 90)
                max_val = np.max(percentiles_max)
            except:
                max_val = -np.inf
            try:
                percentiles_min = region_percentile(patch_img, cellseg_img, 10)
                min_val = np.min(percentiles_min)
            except:
                min_val = np.inf

            max_dict[channel].append(float(max_val))
            min_dict[channel].append(float(min_val))

    min_dict = {k: min(v) for k, v in min_dict.items()}
    max_dict = {k: max(v) for k, v in max_dict.items()}

    # Save dictionary to JSON file
    with open(save_path_min, "w") as f:
        json.dump(min_dict, f)

    # Save dictionary to JSON file
    with open(save_path_max, "w") as f:
        json.dump(max_dict, f)

    return None


def init_worker(patch_img_norm_):
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
    global patch_img_norm
    ROI = patch_img_norm[max(0, bounds_topleft_y-y_buffer):
                            min(bounds_topleft_y+bounds_height+y_buffer, patch_img_norm.shape[0]-1),
                            max(0, bounds_topleft_x-x_buffer):
                            min(bounds_topleft_x+bounds_width+x_buffer, patch_img_norm.shape[1]-1)]
    return ROI, int(label)


def PhenoBIC_predict(base_dir, channels, model_path, num_cells_batch=4000, buffer_ratio=0.1, out_dir_path=None):

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    out_dir_path = os.path.join(base_dir, 'PhenoBIC_output')
    os.makedirs(out_dir_path, exist_ok=True)
    class_save_path = os.path.join(out_dir_path, 'PhenoBIC_cell_phenotype_classes.csv')

    with open(os.path.join(base_dir, 'PhenoBIC_output', 'normalization_max_values.json'), "r") as f:
        max_dict = json.load(f)
    with open(os.path.join(base_dir, 'PhenoBIC_output', 'normalization_min_values.json'), "r") as f:
        min_dict = json.load(f)

    class_dict = {key: [] for key in channels + ['Patch', 'Cell_label']}
    probs_dict = {key: [] for key in channels + ['Patch', 'Cell_label']}

    measurements_save_path = os.path.join(base_dir, "cell_bounds.csv")
    measurements_df = pd.DataFrame({'label': [], 'Bounds_x': [], 'Bounds_y': [], 'Bounds_width': [], 'Bounds_height': []})

    i_patch = 0
    for patch in os.listdir(os.path.join(base_dir, 'multiplex_images')):
        i_patch += 1
        print(f"Processing patch {i_patch} / {len(os.listdir(os.path.join(base_dir, 'multiplex_images')))}")

        cellseg_path = os.path.join(os.path.join(base_dir, 'cell_segmentations', patch + '.tiff'))
        cellseg_img = io.imread(cellseg_path)

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

        num_cells = len(np.unique(cellseg_img)) - 1
        num_iters = (num_cells // num_cells_batch) + (num_cells % num_cells_batch != 0)
        for channel in channels:
            print(patch, channel)
            patch_path = os.path.join(base_dir, 'multiplex_images', patch, channel + '.tif')
            patch_img = io.imread(patch_path)

            min_int = min_dict[channel]
            max_int = max_dict[channel]

            patch_img_norm = np.stack((np.uint8(np.clip((patch_img - min_int) / (max_int - min_int), 0, 1) * 255),) * 3, axis=-1)

            predictions_list = []
            probabilities_list = []

            for i in range(num_iters):

                start_idx = i * num_cells_batch
                end_idx = min(start_idx + num_cells_batch, num_cells)
                arg_tuples = list(zip(
                    measurements_bounds_df['label'].iloc[start_idx:end_idx],
                    (np.floor(measurements_bounds_df['Bounds_x'].iloc[start_idx:end_idx])).astype(int),
                    (np.floor(measurements_bounds_df['Bounds_y'].iloc[start_idx:end_idx])).astype(int),
                    (np.ceil(measurements_bounds_df['Bounds_width'].iloc[start_idx:end_idx])).astype(int),
                    (np.ceil(measurements_bounds_df['Bounds_height'].iloc[start_idx:end_idx])).astype(int),
                    (np.ceil(measurements_bounds_df['Bounds_width'].iloc[start_idx:end_idx] * buffer_ratio)).astype(int),
                    (np.ceil(measurements_bounds_df['Bounds_height'].iloc[start_idx:end_idx] * buffer_ratio)).astype(int)
                    ))
                with mp.Pool(initializer=init_worker, initargs=(patch_img_norm,)) as pool:
                    results = pool.starmap(cell_ROI_channel_extractor, arg_tuples)
                
                preprocessed_ROIs, labels = zip(*results)
                preprocessed_ROIs = list(preprocessed_ROIs)
                labels = list(labels)

                data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
                preprocessed_ROIs = np.array([Image.fromarray(img).resize((48,48), Image.NEAREST) for img in preprocessed_ROIs])
                predict_generator = data_gen.flow(
                    x=preprocessed_ROIs,
                    batch_size=32,
                    shuffle=False
                )

                pred = model.predict(predict_generator)
                pred = pred.reshape(-1)
                predictions= np.empty(len(pred), dtype='<U4')
                predictions[pred <= 0.5] = 'neg'
                predictions[pred > 0.5] = 'pos'
                predictions_list.extend(predictions)
                print(len(predictions_list))
                probabilities_list.extend(pred)
                
            class_dict[channel].extend(predictions_list)

    class_df = pd.DataFrame(class_dict)

    class_df.to_csv(class_save_path, index_label=False)
    measurements_df.to_csv(measurements_save_path, index_label=False)

    return None


def run_PhenoBIC(
        base_dir,
        channels,
        model_path,
        num_cells_batch=4000,
        buffer_ratio=0.1,
        out_dir_path=None):
    
    PhenoBIC_normalize(base_dir=base_dir,
                        channels=channels,
                        out_dir_path=None)
    PhenoBIC_predict(base_dir=base_dir,
                        channels=channels,
                        model_path=model_path,
                        num_cells_batch=num_cells_batch,
                        buffer_ratio=buffer_ratio,
                        out_dir_path=out_dir_path)
    
    return None


if __name__ == "__main__":
    main()