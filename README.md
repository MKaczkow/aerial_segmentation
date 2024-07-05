# Aerial Segmentation
Repo for TWM (Machine Vision Techniques) project @ WUT 24L semester

<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

## Introduction
The following instructions describe:
1. Setting up the environment and running notebooks
2. Functions and purpose of the notebooks
3. Sources and brief description of the data

## Setting up the environment and running notebooks
1. Create a virtual environment (conda, venv, etc.), e.g.: `python -m venv aerial_images`
2. Activate the virtual environment (depending on the operating system):
    - Windows: `.\aerial_images\Scripts\activate`
    - Linux: `source aerial_images/bin/activate`
3. Download and install the `torch` library according to the instructions on [pytorch.org](https://pytorch.org/get-started/locally/)
4. Install the remaining required libraries: `pip install -r requirements.txt`

## Code and notebooks
Generally, the repository is organized in such a way that the `data` directory contains only raw data, the `src` directory contains code and helper functions, and the `notebooks` directory contains notebooks with code. For better clarity, the notebooks are not stored in the root directory, but should be moved there before running. The individual sub-directories contain what their names indicate, below is a detailed description:

* `data` - directory containing only data, directories ending with `Patches` contain datasets divided into patches
* `src` - directory containing source code, including helper functions
    * `callbacks` - directory containing callback functions, assisting in model training management
    * `datasets` - directory containing dataset classes (inheriting from `torch.utils.data.Dataset`)
        * `utils` - directory containing helper functions for datasets, mainly for converting masks to labels of type {0, 1, 2, ...} and transformations (`torchvision.transforms`)
    * `evaluation` - directory containing helper functions for model evaluation
    * `models` - directory containing a helper baseline model class (inheriting from `torch.nn.Module`)
    * `utils.py` - general helper functions
* `notebooks` - directory containing notebooks with code
    * `datasets_to_patches` - notebooks demonstrating the division of datasets into patches
    * `masks_conversion` - notebooks demonstrating the conversion of masks to labels of type {0, 1, 2, ...}
    * `no_finetune` - notebooks demonstrating attempts to use models without training for aerial image segmentation (baseline, weights trained on ImageNet do not transfer to the new dataset)
    * `sanity_checks` - notebooks demonstrating sanity checks for datasets
    * `with_finetune` - notebooks demonstrating actual training of models on new datasets


## Data

### Downloading
* `INRIA`: [link](https://project.inria.fr/aerialimagelabeling/)
* `Dubai`: [link](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery/data)
* `Aerial Drone`: [link](https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset/data)
* `UAVid`: [link](https://www.kaggle.com/code/alexalex02/semantic-segmentation-of-aerial-images)

### Number of classes in datasets
* `INRIA`: 2 (binary - *building* i *non-building*) [source](https://project.inria.fr/aerialimagelabeling/)
* `Dubai`: 6  [source](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery/data)
    - Building: #3C1098
    - Land (unpaved area): #8429F6
    - Road: #6EC1E4
    - Vegetation: #FEDD3A
    - Water: #E2A929
    - Unlabeled: #9B9B9B     
* `Aerial Drone`: 20 (tree, gras, other vegetation, dirt, gravel, rocks, water, paved area, pool, person, dog, car, bicycle, roof, wall, fence, fence-pole, window, door, obstacle) [source](https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset/data)   

> [!WARNING]  
> It looks like, there are actually 23 classes.

* `UAVid`: 8 [source](https://www.kaggle.com/code/alexalex02/semantic-segmentation-of-aerial-images)
    1. *building*: living houses, garages, skyscrapers, security booths, and buildings under construction.
    2. *road*: road or bridge surface that cars can run on legally. Parking lots are not included.
    3. *tree*: tall trees that have canopies and main trunks.
    4. *low vegetation*: grass, bushes and shrubs.
    5. *static car*: cars that are not moving, including static buses, trucks, automobiles, and tractors. Bicycles and motorcycles are not included.
    6. *moving car*: cars that are moving, including moving buses, trucks, automobiles, and tractors. Bicycles and motorcycles are not included.
    7. *human*: pedestrians, bikers, and all other humans occupied by different activities.
    8. *clutter*: all objects not belonging to any of the classes above.

### Number of images in datasets
* `INRIA`
    - `train`: 180 (labels present, need to manually split into train and val)
    - `test`: 144 (no labels)
* `Dubai`
    - `train`: 72 (labels present, need to manually split into train and val)
* `Aerial Drone`
    - `train`: 400 (labels present, need to manually split into train and val)
* `UAVid`
    - `train`: 200
    - `val`: 70
    - `test`: 10

## Resources
* repos
    * [praca z INRIA dataset](https://github.com/margokhokhlova/aerial_segmentation)
    * [praca z Aerial Image Segmentation from Online Maps](https://github.com/alpemek/aerial-segmentation/tree/master)
* datasets
    * [UAVid Semantic Segmentation Dataset](https://www.kaggle.com/datasets/titan15555/uavid-semantic-segmentation-dataset)
    * [Aerial Semantic Segmentation Drone Dataset](https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset)
    * [Semantic segmentation of aerial imagery](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery)
* kaggle solutions
    * [rozwiązanie UAVid Semantic Segmentation Dataset](https://www.kaggle.com/code/alexalex02/semantic-segmentation-of-aerial-images/notebook)
* libraries
    * [potężna libka z gotowymi modelami do segmentacji](https://github.com/qubvel/segmentation_models.pytorch)
* papers
    * [Learning Aerial Image Segmentation from Online Maps](https://ethz.ch/content/dam/ethz/special-interest/baug/igp/photogrammetry-remote-sensing-dam/documents/pdf/Papers/Learning%20Aerial%20Image.pdf)
    * [Drone Depth and Obstacle Segmentation Dataset](https://arxiv.org/pdf/2312.12494.pdf)
    * [Varied Drone Dataset for Semantic Segmentation](https://arxiv.org/pdf/2305.13608.pdf)
