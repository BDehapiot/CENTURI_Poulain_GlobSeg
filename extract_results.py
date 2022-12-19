#%% Imports

import napari
import numpy as np
from skimage import io 
from pathlib import Path
from stardist.models import StarDist2D

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
stack = io.imread(stack_path)

#%%

from skimage.measure import regionprops

properties = []
for t, frame in enumerate(stack):   
    prop = regionprops(frame) 
    ctrd = [p.centroid for p in prop]
    pass
