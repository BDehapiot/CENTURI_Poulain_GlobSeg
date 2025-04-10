#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from skimage import io 
from pathlib import Path
from stardist.models import StarDist2D
from skimage.measure import regionprops
from functions import process_data, norm_data

#%% Inputs --------------------------------------------------------------------

# Select stack
stack_name = 'ML30_outlet_Pdrepano_Temp23_x20_DeltaP20mBars_3_substack(1-1000-10).tif'
# stack_name = 'ML30_outlet_Pdrepano_Temp23_x20_DeltaP20mBars_4_substack(1-1000-10).tif'
# stack_name = 'ML60_inlett_DonneurFYY_Temp23_x20_DeltaP20mBars_3_substack(1-1000-10).tif'
# stack_name = 'ML60_inlett_DonneurFYY_Temp23_x20_DeltaP20mBars_4_substack(1-1000-10).tif'
# stack_name = 'ML60_inlett_Pdrepano_Temp23_x20_DeltaP20mBars_2_substack(1-1000-10).tif'
# stack_name = 'ML60_inlett_Pdrepano_Temp23_x20_DeltaP20mBars_3_substack(1-1000-10).tif'
# stack_name = 'ML60_outlet_DonneurFYY_Temp23_x20_DeltaP20mBars_5_substack(1-1000-10).tif'
# stack_name = 'ML60_outlet_DonneurFYY_Temp23_x20_DeltaP20mBars_6_substack(1-1000-10).tif'

# Parameters
radius = 0 # radius for rolling ball background subtration (0 = deactivate)
display = 1 # display segmentation results in Napari

#%% Initialize ----------------------------------------------------------------

# Paths
data_path = Path(Path.cwd(), 'data', 'raw')
stack_path = data_path / stack_name
model = StarDist2D(None, name='stardist', basedir='models')

#%% Function(s) ---------------------------------------------------------------

def predict(stack):
    
    labels = []
    for t, frame in enumerate(stack):
        tmp, details = model.predict_instances(stack[t,...])   
        labels.append(tmp)
    labels = np.stack(labels)
    
    return labels

def get_properties(labels):
    
    properties = []
    for t, img in enumerate(labels):   
        props = regionprops(img) 
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
    
    return properties

def get_maps(labels, properties):
    
    eccentricity = []
    for t in range(labels.shape[0]):  
        frame = labels[t,...].copy()
        frame = frame.astype('float')
        props = properties[properties[:,0] == t]
        for label in range(props.shape[0]):
            frame[frame==label+1] = props[int(label),7]
        eccentricity.append(frame)
    eccentricity = np.stack(eccentricity)

    orientation = []
    for t in range(labels.shape[0]):  
        frame = labels[t,...].copy()
        frame = frame.astype('float')
        props = properties[properties[:,0] == t]
        for label in range(props.shape[0]):
            frame[frame==label+1] = props[int(label),8]
        orientation.append(frame)
    orientation = np.stack(orientation)
    
    return eccentricity, orientation    

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Open & process
    stack = io.imread(stack_path)
    stack = process_data(stack, radius=radius)
    stack = norm_data(stack, qlow=0.001, qhigh=0.999)
    
    # Predict
    labels = predict(stack)
    
    # Get properties
    properties = get_properties(labels)
    
    # Get maps
    eccentricity, orientation = get_maps(labels, properties)
    
    # Save
    io.imsave(
        data_path / (stack_name.replace(".tif", "_predict.tif")),
        labels.astype("uint16"), check_contrast=False,
        )
    np.savetxt(
        data_path / (stack_name.replace(".tif", "_predict.csv")), 
        properties, delimiter=",", fmt='%10.5f', 
        header=(
            'frame,label,area,'
            'ctrd_x,ctrd_y,'
            'maj_axis,min_axis,'
            'eccentricity,orientation'
            )
        )
    
    # Display
    if display:
        viewer = napari.Viewer()
        viewer.add_image(orientation, visible=0)
        viewer.add_image(eccentricity, visible=0)
        viewer.add_labels(labels, visible=1)
        viewer.add_image(stack, opacity=0.5, blending="additive", visible=1)

