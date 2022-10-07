#%% Imports

import napari
import numpy as np
from skimage import io 
from pathlib import Path
from stardist.models import StarDist2D

from functions import process_data, norm_data

#%% Inputs

radius = 0 # radius for rolling ball background subtration (0 = deactivate)

#%% Get stack_name

# stack_name = '20mbar_end_AV_20171019_162451_substack(1-900-5).tif'
# stack_name = '35mbar_end_05_20170613_151622_substack(1-900-5).tif'
# stack_name = '50mbar_start_00_20170613_111653_substack(1-900-5).tif'

stack_name = 'ML30__inlet_donneurFYY_Temp267_x20_DeltaP20mBars_vid0002.tif'
# stack_name = 'ML30_outlet_donneurCJJ_Temp267_x20_DeltaP50mBars_vid0001.tif'
# stack_name = 'ML30_outlet_donneurFYY_Temp267_x20_DeltaP20mBars_vid0002.tif'

#%% Paths & import

stack_path = Path(Path.cwd(), 'data', 'raw', stack_name)
stack = io.imread(stack_path)
model = StarDist2D(None, name='stardist', basedir='models')

#%% Predict

stack = process_data(stack, radius=radius, parallel=True)
stack = norm_data(stack, qlow=0.001, qhigh=0.999)

# frame
# t = 100
# labels, details = model.predict_instances(stack[t,...])

# stack
tmin = 0
tmax = 100
labels = np.zeros((tmax, stack.shape[1], stack.shape[2]))
for t in range(tmin, tmax):
    labels[t,...], details = model.predict_instances(stack[t,...])    

#%% Save

io.imsave(stack_path.stem + '_predict.tif' , labels.astype('uint16'), check_contrast=False)

#%% Display

# viewer = napari.Viewer()
# viewer.add_image(stack[t,...])
# viewer.add_labels(labels)
# viewer.grid.enabled = True