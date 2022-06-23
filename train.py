#%% Imports

import napari
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
        if 'crop' in path.stem:       
            if 'labels' in path.stem:
                Y_paths.append(path)
                Y.append(io.imread(path))
            else:
                X_paths.append(path)
                X.append(io.imread(path))
                               
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
    
i = min(9, len(X)-1)
img, lbl = X[i], Y[i]
assert img.ndim in (2,3)
img = img if (img.ndim==2 or img.shape[-1]==3) else img[...,0]
plot_img_label(img,lbl)
None;    

# X and Y do not correspond!
i = 3
plot_img_label(X[i],Y[i])


