#%% Imports

import napari
import numpy as np
from skimage import io 
import tensorflow as tf
from pathlib import Path
import albumentations as aug
import segmentation_models as sm

from functions import norm_data, split_data, augment_data, process_data

sm.set_framework('tf.keras')
sm.framework()

#%% Initialize

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#%% Paths & import

for path in Path('data', 'train').iterdir(): 

    if 'crop' in path.name and 'mask' in path.name:   
            
        raw_path = Path('data', 'train', path.name.replace('_mask', ''))
        
        if 'raw' not in globals(): 
            raw = io.imread(raw_path)
            mask = io.imread(path)   
            
        else:
            raw = np.dstack((raw, io.imread(raw_path)))
            mask = np.dstack((mask, io.imread(path))) 
            
raw = np.rollaxis(raw, 2)      
mask = np.rollaxis(mask, 2) 

#%%

from skimage.measure import regionprops, regionprops_table
from scipy.ndimage import distance_transform_edt

for t, frame in enumerate(mask):
    
    edm = np.full(frame.shape, 1, dtype=float)    
    
    for prop in regionprops(frame):
        
        y, x = prop.centroid
        y = int(y); x = int(x)
        edm[y,x] = 0
        
    edm = distance_transform_edt(edm)
    edm[frame==0] = 0
    

    

    
    

#%%

# # Normalize data
# raw = norm_data(raw)

# # Split data
# raw_trn, mask_trn, raw_val, mask_val = split_data(raw, mask, 0.33)

# # Augment data
# operations = aug.Compose([
#     aug.VerticalFlip(p=0.5),              
#     aug.RandomRotate90(p=0.5),
#     aug.HorizontalFlip(p=0.5),
#     aug.Transpose(p=0.5),
#     aug.GridDistortion(p=0.5),
#     ])

# raw_trn, mask_trn = augment_data(
#     raw_trn, mask_trn, operations, iterations=raw_trn.shape[0]*10, parallel=False
#     )

# raw_val, mask_val = augment_data(
#     raw_val, mask_val, operations, iterations=raw_val.shape[0]*10, parallel=False
#     )

# # Binarize mask
# mask_trn = (mask_trn > 0).astype('float')
# mask_val = (mask_val > 0).astype('float')

#%% Train

# # Define model
# model = sm.Unet(
#     'resnet34', 
#     input_shape=(None, None, 1), 
#     classes=1, 
#     encoder_weights=None
#     )

# model.compile('Adam',
#     loss=sm.losses.bce_jaccard_loss,
#     metrics=[sm.metrics.iou_score],
# )

# # Fit model
# model.fit(
#    x=raw_trn,
#    y=mask_trn,
#    batch_size=32,
#    epochs=64,
#    validation_data=(raw_val, mask_val),
#    verbose=1,
# )

# tensorboard --logdir /home/bdehapiot/Projects/CENTURI_Poulain_GlobSeg/models/stardist/logs

#%% Predict

# crop_name = 'ML30_outlet_Pdrepano_Temp23_x20_DeltaP20mBars_3_substack(1-1000-10)_crop.tif'
# crop_path = Path(Path.cwd(), 'data', 'raw', crop_name)
# crop = io.imread(crop_path)

# # Process and normalize
# crop = process_data(crop, radius=0, parallel=True)
# crop = norm_data(crop)

# prediction = model.predict(crop).squeeze()

#%% 

viewer = napari.Viewer()
viewer.add_image(edm)