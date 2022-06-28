#%% Imports

import numpy as np
from skimage import io 
from pathlib import Path
from joblib import Parallel, delayed 
from skimage.restoration import rolling_ball

#%% Inputs

sample_size = 50 # number of random cropped images 
seed = 1 # seed for random indexes
crop_size = 128 # size of random cropped images 
radius = 0 # radius for rolling ball background subtration (0 = deactivate)

#%% functions

def range_uint8(img, int_range=0.99):

    ''' 
    Description
    
    Parameters
    ----------
    img : ndarray
        Description        
        
    int_range : float
        Description
    
    Returns
    -------  
    img : ndarray
        Description
        
    Notes
    -----   
    
    '''

    # Get data type 
    data_type = (img.dtype).name
    
    if data_type == 'uint8':
        
        raise ValueError('Input image is already uint8') 
        
    else:
        
        # Get data intensity range
        int_min = np.percentile(img, (1-int_range)*100)
        int_max = np.percentile(img, int_range*100) 
        
        # Rescale data
        img[img<int_min] = int_min
        img[img>int_max] = int_max 
        img = (img - int_min)/(int_max - int_min)
        img = (img*255).astype('uint8')
    
    return img

''' ----------------------------------------------------------------------- '''

def process_data(raw, radius=radius, parallel=True):
    
    """
    Description
    
    Parameters
    ----------
    raw : ndarray
        Description
    radius : int
        Description
    
    Returns
    -------
    process : ndarray
        Description
    
    Raises
    ------
    """
    
    # Nested function ---------------------------------------------------------
    
    def _process_data(temp):
    
        # Subtract background
        process =  temp - rolling_ball(temp, radius=radius)
        
        return process
    
    # Main function -----------------------------------------------------------
        
    # Subtract mean projection
    temp = raw - np.mean(raw, axis=0)
    
    if radius > 0:
    
        if parallel:
     
            # Run _process_data (parallel)
            output_list = Parallel(n_jobs=-1)(
                delayed(_process_data)(
                    img
                    )
                for img in temp
                )
                
        else:
                
            # Run _process_data
            output_list = [_process_data(
                    img
                    ) 
                for img in temp
                ]
    
        # Extract outputs
        process = np.stack([arrays for arrays in output_list], axis=0)
        
    else:   
            
        process = temp
            
    return process
    
#%% Open & process raw data

data = []
for path in Path('data', 'raw').iterdir():   
    raw = io.imread(path)   
    process = process_data(raw, radius=radius, parallel=True)
    data.append((raw, process, raw.shape, path))

#%% Generate random cropped images (seeded)

np.random.seed(seed)

# Generate random indexes
randR = np.random.randint(0, len(data), size=sample_size)
randI = np.zeros_like(randR)
randY = np.zeros_like(randR)
randX = np.zeros_like(randR)
for i in range(randR.shape[0]):    
    randI[i] = np.random.randint(0, data[randR[i]][0].shape[0])
    randY[i] = np.random.randint(0, data[randR[i]][0].shape[1]-crop_size)
    randX[i] = np.random.randint(0, data[randR[i]][0].shape[2]-crop_size)    

#%% Extract & save cropped data

for r, i, y, x in zip(randR, randI, randY, randX):
    crop = data[r][1][i,y:y+crop_size,x:x+crop_size]
    path = Path('data', 'train', 
        f'{data[r][3].stem}_crop({r:02}-{i:04}-{y:04}-{x:04}).tif'
        )    
    io.imsave(path, range_uint8(crop, int_range=0.99), check_contrast=False)
        