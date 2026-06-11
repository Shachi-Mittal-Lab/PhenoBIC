# About

PhenoBIC allows for automated and operator-independent cell biomarker expression classification with spatial multiplex imaging data. PhenoBIC is a deep learning model that predicts marker co-expression phenotype of cells by performing image classification of the biomarker "imprint" (staining pattern) of the cell.

---

## Citing

If you use PhenoBIC, please consider giving this repository a star ⭐️ and also cite the following:

- **PhenoBIC paper**: Sankaranarayanan, A. et al.

---

## System Requirements

- Supports TIFF and TIFF-derived multiplex image formats (e.g. OME-TIFF)

---

## Installation

### Step 1: Install the PhenoBIC Python environment

You need a Python environment with the PhenoBIC dependencies. You can use conda:

1. Install [Anaconda](https://www.anaconda.com/).

2. Download `environment.yml`

3. In an Anaconda prompt terminal, get into the directory with the `environment.yml`
E.g.
```bash
cd C:\path_to_folder_containing_environment.yml 
```
Then create the Python enviornment with the required packages to run PhenoBIC
```bash
conda env create -f environment.yml  
```
Then activate the Python environment:
```bash
conda activate PhenoBIC 
```

It is not necessary to use a GPU for PhenoBIC inference but **for GPU acceleration**- you may need to install a GPU driver if you have not already. And then you need to install CUDA and cuDNN with conda to run tensorflow with GPU on Windows native.

E.g.
```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 
```
Please refer to [tensorflow documentation](https://www.tensorflow.org/install/pip#windows-native_1) for additional instructions.

### Step 2: Download the PhenoBIC model

Install the PhenoBIC `.keras` model from [here](https://github.com/your-org/qupath-extension-phenobic/releases)

---

## Running PhenoBIC

There are three ways to run PhenoBIC

1. QuPath extension
2. Python code → Full multiplex image + tabular cell segmentation data
3. Python code → Patched multiplex data + cell segmentation masks

### QuPath extension

We have provided an [extension to directly implement PhenoBIC in a QuPath project using the interface](https://github.com/Shachi-Mittal-Lab/PhenoBIC-qupath-extension/tree/main).

### Python code → Full multiplex image + tabular cell segmentation data

#### Requirements

- TIFF and TIFF-derived multiplex image format (e.g., OMETIFF)
- Must know the index in TIFF channel stack for each channel (e.g., CD3=channel 0, CD8=channel 1, etc.)
- Cell segmentation output- the bounding box of each cell- must be stored in a CSV file with the following columns:

    | Object ID                              | Bounds_x        | Bounds_y        | Bounds_width        | Bounds_height        |
    |----------------------------------------|-----------------|-----------------|--------------------|--------------------|
    | 6f01e8fb-c6e6-4dd2-a293-16e709f26027   | 3922    | 5000 | 21  | 30  |
    | 7400fe4a-ae25-478c-9ca2-c26fcddecedb   | 3664   | 305 | 15  | 12  |
    | ....   | .... | ....| ....  | ....  |

    <b>Object ID</b> is a unique identifier for each cell. <b>Bounds_x</b> and <b>Bounds_y</b> are the upper left-hand coordinates of each cell's bounding box. <b>Bounds_width</b> is the width (x-dimension) of the bounding box and <b>Bounds-height</b> is the height (y-dimension).

    This reflects the styling of a QuPath measurements export. A [groovy script](google.com) to generate the bounding box measurements for cells in a QuPath project is provided.

    If you are saving cell segmentations in another manner (e.g., masks, etc.)- you can convert them into this tabular format to be compatible with this code. Alternatively, you can use the patched approach, as shown below, that uses cell segmentation masks

#### Implementation
Example notebook shown in `run_PhenoBIC_coordinates.ipynb`.

### Python code → Patched multiplex data + cell segmentation masks

#### Requirements
Required file directory structure of multiplex channels and cell segmentation masks:


    |-{Sample base}
    ||-cell_segmentations
    |||-{Patch_1}.tiff (or .tif)
    |||-{Patch_2}.tiff (or .tif)
    |||-{Patch_3}.tiff (or .tif)
    ||-multiplex_images
    |||-{Patch_1}
    ||||-{Channel_1}.tiff (or .tif)
    ||||-{Channel_2}.tiff (or .tif)
    ||||-{Channel_3}.tiff (or .tif)
    |||-{Patch_2}
    ||||-{Channel_1}.tiff (or .tif)
    ||||-{Channel_2}.tiff (or .tif)
    ||||-{Channel_3}.tiff (or .tif)
    |||-{Patch_3}
    ||||-{Channel_1}.tiff (or .tif)
    ||||-{Channel_2}.tiff (or .tif)
    ||||-{Channel_3}.tiff (or .tif)

Replace {Sample base}, {Patch_#}, and {Channel_#} to match your naming convention for samples, patches/tiles, and marker channels.

Sample data are shown in [XXXX](google.com)

#### Implementation
Example notebook shown in `run_PhenoBIC_masks.ipynb`. Results are stored in `{Sample base}/PhenoBIC_output`

We find that this approach can be slower than the tabular approach that directly reads from a large image, especially for small patch sizes.
