## Usage

<img src='example.png' alt="example">

### `main.py`
Read a selected `TIF` stack from the `data_path` directory and 
segments/measures the detected red blood cells (RBCs). Segmentation is 
performed using a custom-trained StarDist2D model. Labelled masks and
measurments are saved in the `data_path` directory.

- #### Paths
```bash
- data_path   # str, path to the data directory
```

- #### Parameters
```bash
- stack_name  # str, name (with extension) of the stack to process
- radius      # int, radius for rolling ball background subtraction [*] 
- display     # bool, whether to display labelled masks & measurements in Napari
```
```
The radius should be set to 0, as the current model was trained without 
background subtraction.
```

- #### Outputs
```bash
- ..._predict.tif  # TIF, labelled masks saved as uint16
- ..._predict.csv  # CSV, measurments saved in an annotated CSV file
```
```bash
- frame            # frame number for the segmented object
- label            # object label
- area             # object area (pixels)
- ctrd_x           # x-coordinate of the object centroid (pixels)
- ctrd_y           # y-coordinate of the object centroid (pixels)
- maj_axis         # major axis length of fitted ellipse (pixels)
- min_axis         # minor axis length of fitted ellipse (pixels)
- eccentricity     # from 0 (circle) to 1 (line)
- orientation      # angle of the major axis orientation, from -π/2 to π/2
```

### `functions.py`
Contains helper functions required to run `main.py`

### `models`
Includes the custom-trained StarDist2D model, as well as code used to extract
the training data (`model_extract.py`) and train the model (`model_train.py`).
Note that these scripts are no longer maintained and may require adjustments to
run.