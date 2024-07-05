#%% Imports

import napari
import pandas as pd
import numpy as np
from skimage import io 
from pathlib import Path
from skimage.measure import regionprops

from functions import process_data, norm_data

#%% Get stack_name

stack_name = 'ML30_outlet_Pdrepano_Temp23_x20_DeltaP20mBars_3_substack(1-1000-10)_predict.tif'
# stack_name = 'ML30_outlet_Pdrepano_Temp23_x20_DeltaP20mBars_4_substack(1-1000-10)_predict.tif'
# stack_name = 'ML60_inlett_DonneurFYY_Temp23_x20_DeltaP20mBars_3_substack(1-1000-10)_predict.tif'
# stack_name = 'ML60_inlett_DonneurFYY_Temp23_x20_DeltaP20mBars_4_substack(1-1000-10)_predict.tif'
# stack_name = 'ML60_inlett_Pdrepano_Temp23_x20_DeltaP20mBars_2_substack(1-1000-10)_predict.tif'
# stack_name = 'ML60_inlett_Pdrepano_Temp23_x20_DeltaP20mBars_3_substack(1-1000-10)_predict.tif'
# stack_name = 'ML60_outlet_DonneurFYY_Temp23_x20_DeltaP20mBars_5_substack(1-1000-10)_predict.tif'
# stack_name = 'ML60_outlet_DonneurFYY_Temp23_x20_DeltaP20mBars_6_substack(1-1000-10)_predict.tif'

#%% Paths & import

stack_path = Path(Path.cwd(), 'data', 'raw', stack_name)
csv_path = Path(Path.cwd(), 'data', 'raw', stack_name.replace('.tif', '.csv'))
stack = io.imread(stack_path)

#%%

properties = []
for t, frame in enumerate(stack):   
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
np.savetxt(
    csv_path, properties, delimiter=",", fmt='%10.5f', 
    header='frame,label,area,ctrd_x,ctrd_y,maj_axis,min_axis,eccentricity,orientation'
    )
    
#%%

