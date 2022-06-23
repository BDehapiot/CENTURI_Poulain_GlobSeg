#%% Imports

# import napari
import numpy as np
from skimage import io 
from pathlib import Path
import matplotlib.pyplot as plt
from joblib import Parallel, delayed 

#%% Inputs

stack_path = Path('data','20mbar_end_AV_20171019_162451_substack(1-900-5).tif')
# stack_path = Path('data','35mbar_end_05_20170613_151622_substack(1-900-5).tif')

#%% Initialize

# Create paths & open data
X_paths = []; X = []
Y_paths = []; Y = []
for path in Path('data').iterdir():      
    if stack_path.stem in path.stem:
        if 'crop' in path.stem and 'labels' not in path.stem: 
            X_paths.append(path)
            X.append(io.imread(X_paths[-1]))            
            Y_paths.append(Path('data', path.stem + '_labels.tif'))
            Y.append(io.imread(Y_paths[-1]))
                               
#%% Fitting ground-truth labels with star-convex polygons

# from tqdm import tqdm
# from stardist import relabel_image_stardist, random_label_cmap
# from stardist.matching import matching_dataset

# np.random.seed(42)
# lbl_cmap = random_label_cmap()

# n_rays = [2**i for i in range(2,8)]
# scores = []
# for r in tqdm(n_rays):
#     Y_reconstructed = [relabel_image_stardist(lbl, n_rays=r) for lbl in Y]
#     mean_iou = matching_dataset(Y, Y_reconstructed, thresh=0, show_progress=False).mean_true_score
#     scores.append(mean_iou)

# plt.figure(figsize=(8,5))
# plt.plot(n_rays, scores, 'o-')
# plt.xlabel('Number of rays for star-convex polygon')
# plt.ylabel('Reconstruction score (mean intersection over union)')
# plt.title("Accuracy of ground truth reconstruction (should be > 0.8 for a reasonable number of rays)")
# None;

#%% Example image reconstructed with various number of rays

# fig, ax = plt.subplots(2,3, figsize=(16,11))
# for a,r in zip(ax.flat,n_rays):
#     a.imshow(relabel_image_stardist(Y[0], n_rays=r), cmap=lbl_cmap)
#     a.set_title('Reconstructed (%d rays)' % r)
#     a.axis('off')
# plt.tight_layout();

#%%

from tqdm import tqdm
from csbdeep.utils import normalize
from stardist import fill_label_holes, random_label_cmap

np.random.seed(42)
lbl_cmap = random_label_cmap()

n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]

# Normalize images and fill small label holes
axis_norm = (0,1)
X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X)]
Y = [fill_label_holes(y) for y in tqdm(Y)]

# Split into train and validation datasets
rng = np.random.RandomState(42)
ind = rng.permutation(len(X))
n_val = max(1, int(round(0.25 * len(ind))))
ind_train, ind_val = ind[:-n_val], ind[-n_val:]
X_val, Y_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val]
X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train] 
print('number of images: %3d' % len(X))
print('- training:       %3d' % len(X_trn))
print('- validation:     %3d' % len(X_val))

def plot_img_label(img, lbl, img_title="image", lbl_title="label", **kwargs):
    fig, (ai,al) = plt.subplots(1,2, figsize=(12,5), gridspec_kw=dict(width_ratios=(1.25,1)))
    im = ai.imshow(img, cmap='gray', clim=(0,1))
    ai.set_title(img_title)    
    fig.colorbar(im, ax=ai)
    al.imshow(lbl, cmap=lbl_cmap)
    al.set_title(lbl_title)
    plt.tight_layout()
    
# i = min(9, len(X)-1)
# img, lbl = X[i], Y[i]
# assert img.ndim in (2,3)
# img = img if (img.ndim==2 or img.shape[-1]==3) else img[...,0]
# plot_img_label(img,lbl)
# None;    

#%%

from stardist import gputools_available, calculate_extents
from stardist.models import Config2D, StarDist2D

conf = Config2D (
    n_rays=32,
    grid=(2,2),
    use_gpu=True and gputools_available(),
    n_channel_in=1,
    train_epochs=50,
    train_patch_size=(128,128),
    )

model = StarDist2D(conf, name='stardist', basedir='models')

# median_size = calculate_extents(list(Y), np.median)
# fov = np.array(model._axes_tile_overlap('YX'))
# print(f"median object size:      {median_size}")
# print(f"network field of view :  {fov}")
# if any(median_size > fov):
#     print("WARNING: median object size larger than field of view of the neural network.")

#%%

# import napari
import random
import albumentations as A

# Data augmentation

def _data_augmentation(raw, mask, operations):
    
    rand = random.randint(0, raw.shape[0]-1)
    outputs = operations(image=raw[rand,:,:], mask=mask[rand,:,:])
        
    raw_aug = outputs['image']
    mask_aug = outputs['mask']
    
    return raw_aug, mask_aug

''' ........................................................................'''

def data_augmentation(raw, mask, operations, iterations=256, parallel=True):
    
    if type(raw) == list:    
        islist = 1
        raw = np.rollaxis(np.dstack(raw),-1)
        mask = np.rollaxis(np.dstack(mask),-1)
    else:
        islist = 0
    
    if parallel:
 
        # Run _data_augmentation (parallel)
        output_list = Parallel(n_jobs=-1)(
            delayed(_data_augmentation)(
                raw,
                mask,
                operations
                )
            for i in range(iterations)
            )
            
    else:
            
        # Run _data_augmentation
        output_list = [_data_augmentation(
                raw,
                mask,
                operations
                ) 
            for i in range(iterations)
            ]

    # Extract outputs
    if islist == 0:
        raw_augmented = np.stack(
            [arrays[0] for arrays in output_list],
            axis=0
            )
        mask_augmented = np.stack(
            [arrays[1] for arrays in output_list],
            axis=0
            )
    else:
        raw_augmented = [i[0] for i in output_list]
        mask_augmented = [i[1] for i in output_list]
    
    return raw_augmented, mask_augmented

# Define operations 
operations = A.Compose([
    A.VerticalFlip(p=0.5),              
    # A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    # A.Transpose(p=0.5),
    A.GridDistortion(p=0.5)
    ]
)

X_trn_aug, Y_trn_aug = data_augmentation(
    X_trn, Y_trn, operations, iterations=1000, parallel=False)

#%%

model.train(X_trn_aug, Y_trn_aug, validation_data=(X_val,Y_val), augmenter=None)


#%%

Y_val_pred = [model.predict_instances(
    x, n_tiles=model._guess_n_tiles(x), 
    show_tile_progress=False)[0]
              for x in tqdm(X_val)]

plot_img_label(X_val[0],Y_val[0], lbl_title="label GT")
plot_img_label(X_val[0],Y_val_pred[0], lbl_title="label Pred")
