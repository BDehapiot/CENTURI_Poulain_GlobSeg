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

stack_name = 'ML30_outlet_Pdrepano_Temp23_x20_DeltaP20mBars_3_substack(1-1000-10).tif'

#%% Paths & import

stack_path = Path(Path.cwd(), 'data', 'raw', stack_name)
stack = io.imread(stack_path)
model = StarDist2D(None, name='stardist', basedir='models')

#%% Predict

stack = process_data(stack, radius=radius, parallel=True)
stack = norm_data(stack, qlow=0.001, qhigh=0.999)

# # stack
# tmin = 0; tmax = 100
# labels = np.zeros((tmax, stack.shape[1], stack.shape[2]))
# for t in range(tmin, tmax):
#     labels[t,...], details = model.predict_instances(stack[t,...])   

#%%  

from stardist.geometry import _dist_to_coord_old as dist_to_coord
from stardist.nms import _non_maximum_suppression_old as non_maximum_suppression
from stardist.geometry import _polygons_to_label_old as polygons_to_label
from stardist import random_label_cmap, draw_polygons, _draw_polygons, sample_points
from stardist.models import Config2D, StarDist2D

prob, dist = model.predict(stack[0,...])
coord = dist_to_coord(dist, grid=model.config.grid)
points = non_maximum_suppression(coord, prob, prob_thresh=0.25, nms_thresh=0.4, grid=model.config.grid)
labels = polygons_to_label(coord, prob, points, shape=stack[0,...].shape)

#%% Save

# io.imsave(
#     Path('data/raw/', stack_name.replace('.tif', '_predict.tif')),
#     labels.astype('uint16'), 
#     check_contrast=False
#     )

io.imsave(
    Path('data/raw/', stack_name.replace('.tif', '_prob.tif')),
    prob, 
    check_contrast=False
    )

#%% Display

# from skimage.transform import resize

viewer = napari.Viewer()
viewer.add_image(stack[0,...])
viewer.add_labels(labels)
# viewer.add_image(resize(prob, stack[0,...].shape, preserve_range=True))
# viewer.add_image(np.moveaxis(dist, 2, 0))
# viewer.grid.enabled = True