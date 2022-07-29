#%% Imports

import napari
import numpy as np
from skimage import io 
from pathlib import Path
from stardist.models import StarDist2D

from functions import process_data, norm_data

#%% Get stack_name

stack_name = '20mbar_end_AV_20171019_162451_substack(1-900-5).tif'
# stack_name = '35mbar_end_05_20170613_151622_substack(1-900-5).tif'
# stack_name = '50mbar_start_00_20170613_111653_substack(1-900-5).tif'

#%% Paths & import

stack = io.imread(Path(Path.cwd(), 'data', 'raw', stack_name))
model = StarDist2D(None, name='stardist', basedir='models')

#%% Predict

stack = process_data(stack, radius=3, parallel=True)
stack = norm_data(stack, qlow=0.001, qhigh=0.999)
labels, details = model.predict_instances(stack[0,...])

#%% Display

viewer = napari.Viewer()
viewer.add_image(stack[0,...])
viewer.add_labels(labels)
# viewer.grid.enabled = True
