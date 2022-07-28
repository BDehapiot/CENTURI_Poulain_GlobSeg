#%% Imports

import napari
import numpy as np
from skimage import io 
from pathlib import Path
from stardist.models import StarDist2D

#%% Get img_name

img_name = '20mbar_end_AV_20171019_162451_substack(1-900-5)_t0.tif'

#%% Paths & import

img = io.imread(Path(Path.cwd(), 'data', img_name))
model = StarDist2D(None, name='stardist', basedir='models')

#%% Predict

labels, details = model.predict_instances(img)

#%% Display

viewer = napari.Viewer()
viewer.add_image(img)
viewer.add_labels(labels)
viewer.grid.enabled = True
