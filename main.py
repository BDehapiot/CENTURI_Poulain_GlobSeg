#%% Imports

import napari
import numpy as np
from skimage import io 
from pathlib import Path
from skimage.transform import resize
from stardist.models import StarDist2D
from skimage.measure import regionprops

from functions import process_data, norm_data

#%% Inputs

radius = 0 # radius for rolling ball background subtration (0 = deactivate)

#%% Get stack_name

# stack_name = 'ML30_outlet_Pdrepano_Temp23_x20_DeltaP20mBars_3_substack(1-1000-10).tif'
stack_name = 'ML30_outlet_Pdrepano_Temp23_x20_DeltaP20mBars_4_substack(1-1000-10).tif'
# stack_name = 'ML60_inlett_DonneurFYY_Temp23_x20_DeltaP20mBars_3_substack(1-1000-10).tif'
# stack_name = 'ML60_inlett_DonneurFYY_Temp23_x20_DeltaP20mBars_4_substack(1-1000-10).tif'
# stack_name = 'ML60_inlett_Pdrepano_Temp23_x20_DeltaP20mBars_2_substack(1-1000-10).tif'
# stack_name = 'ML60_inlett_Pdrepano_Temp23_x20_DeltaP20mBars_3_substack(1-1000-10).tif'
# stack_name = 'ML60_outlet_DonneurFYY_Temp23_x20_DeltaP20mBars_5_substack(1-1000-10).tif'
# stack_name = 'ML60_outlet_DonneurFYY_Temp23_x20_DeltaP20mBars_6_substack(1-1000-10).tif'

#%% Paths & import

stack_path = Path(Path.cwd(), 'data', 'raw', stack_name)
perdict_path = Path(Path.cwd(), 'data', 'raw', stack_name.replace('.tif', '_predict.tif'))
csv_path = Path(Path.cwd(), 'data', 'raw', stack_name.replace('.tif', '_predict.csv'))
stack = io.imread(stack_path)
model = StarDist2D(None, name='stardist', basedir='models')

#%% Prediction

# Process and normalize
stack = process_data(stack, radius=radius, parallel=True)
stack = norm_data(stack, qlow=0.001, qhigh=0.999)

# Predict
labels = []
for t, frame in enumerate(stack):
    temp, details = model.predict_instances(stack[t,...])   
    labels.append(temp)
labels = np.array(labels)

#%% Properties

# Extract
properties = []
for t, frame in enumerate(labels):   
    props = regionprops(frame) 
    properties.append(np.column_stack((
        np.full(len(props), t),
        np.array([p.label for p in props]), 
        np.array([p.area for p in props]), 
        np.array([p.centroid for p in props]),
        np.array([p.axis_major_length for p in props]),
        np.array([p.axis_minor_length for p in props]),
        np.array([p.eccentricity for p in props]),
        np.array([p.orientation for p in props]),
        )))
properties = np.vstack(properties)

# Save
np.savetxt(
    csv_path, properties, delimiter=",", fmt='%10.5f', 
    header='frame,label,area,ctrd_x,ctrd_y,maj_axis,min_axis,eccentricity,orientation'
    )

#%% Maps

eccentricity = []
for t in range(stack.shape[0]):  
    frame = labels[t,...].copy()
    frame = frame.astype('float')
    props = properties[properties[:,0] == t]
    for label in range(props.shape[0]):
        frame[frame==label+1] = props[int(label),7] # eccentricity
    eccentricity.append(frame)
eccentricity = np.stack(eccentricity)

orientation = []
for t in range(stack.shape[0]):  
    frame = labels[t,...].copy()
    frame = frame.astype('float')
    props = properties[properties[:,0] == t]
    for label in range(props.shape[0]):
        frame[frame==label+1] = props[int(label),8] # orientation
    orientation.append(frame)
orientation = np.stack(orientation)

#%% Display

viewer = napari.Viewer()
viewer.add_image(stack)
viewer.add_labels(labels)
viewer.add_image(eccentricity)
viewer.add_image(orientation)

#%% Save

# io.imsave(
#     Path('data/raw/', stack_name.replace('.tif', '_predict.tif')),
#     labels.astype('uint16'), 
#     check_contrast=False
#     )
