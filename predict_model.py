#%% Imports

import napari
import numpy as np
from skimage import io 
from pathlib import Path
from skimage.transform import resize
from stardist.models import StarDist2D

from functions import process_data, norm_data

#%% Inputs

radius = 0 # radius for rolling ball background subtration (0 = deactivate)

#%% Get stack_name

stack_name = 'ML30_outlet_Pdrepano_Temp23_x20_DeltaP20mBars_3_substack(1-1000-10).tif'

#%% Paths & import

stack_path = Path(Path.cwd(), 'data', 'raw', stack_name)
stack = io.imread(stack_path)
model = StarDist2D(None, name='stardist', basedir='models')

#%% Predict

# Process and normalize
stack = process_data(stack, radius=radius, parallel=True)
stack = norm_data(stack, qlow=0.001, qhigh=0.999)

# Labels
labels = []
for t, frame in enumerate(stack):
    temp, details = model.predict_instances(stack[t,...])   
    labels.append(temp)
labels = np.array(labels)

# Prob
prob = []
for t, frame in enumerate(stack):
    temp, _ =  model.predict(stack[t,...]) 
    prob.append(temp) 
prob = np.array(prob)

#%% Process predictions

from skimage.measure import label

prob = resize(prob, stack.shape, preserve_range=True)
markers = np.zeros_like(prob, dtype='uint16')
for t, temp in enumerate(prob):
    markers[t,...] = label(temp>0.5) 

#%% Display

# from skimage.transform import resize

viewer = napari.Viewer()
viewer.add_image(stack)
viewer.add_labels(labels)
# viewer.add_image(prob)
viewer.add_image(markers>0)
# viewer.grid.enabled = True

#%% Save

# io.imsave(
#     Path('data/raw/', stack_name.replace('.tif', '_predict.tif')),
#     labels.astype('uint16'), 
#     check_contrast=False
#     )

# io.imsave(
#     Path('data/raw/', stack_name.replace('.tif', '_prob.tif')),
#     prob, 
#     check_contrast=False
#     )
