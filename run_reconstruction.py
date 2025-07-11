import sys
sys.path.append('../')
sys.path.append('../../')
from argeoPET import array_lib as np
import numpy
import optimization_algorithms as algos
#np.cuda.Device(3).use()
#EVERYTHING that goes into the projector MUST BE FLOAT32


######### INPUT
#### DATA
mouse_path = "data/mouse_background_many.npy"; mouse_down_factor = int(1) #8
plaque_path = "data/plaque.npy"; plaque_down_factor = int(1) #16    

#### SCANNER FOV size (in same units as data)
radius = 17.; half_axial_length = 22.05 
# Creates an extruded square for the reconstruction where a cylinder of radius "radius" and length "half_axial_length" is inscribed
# Data points are taken wrt to this. origins is at -r and -half_axial_length.

#### Reconstruction parameters
crop = np.s_[10:390,60:260,30:270] #wrt to PT1, which is (402,310,310) [2hal, 2r, 2r] #check mp_reconstruction.npy for where to crop
Ns_axial = 90

#### This scales the whole thing if one wants to crop, to tidy up after further testing
if True:
    og_vol = (402,310,310)
    og_voxel_size = (2*np.array([half_axial_length, radius, radius])/np.array(og_vol)).astype(np.float32)
    og_origin = ((-np.array(og_vol)/ 2 + 0.5) * og_voxel_size).astype(np.float32)

    
    starts = np.array([s.start for s in crop])
    origin = (og_origin + starts*og_voxel_size).astype(np.float32)

    sizes = np.array([s.stop - s.start for s in crop])
    new_lengths = sizes*og_voxel_size
    #if no change:
    #vol_shape  = sizes
    #voxel_size = og_voxel_size
    vol_shape = np.rint(sizes/(sizes[0]/Ns_axial)).astype(int)
    voxel_size = (new_lengths/vol_shape).astype(np.float32)
    vol_shape = tuple(vol_shape.tolist())

nsubsets = 30 # use 1/30th of the data at a time for the OSEM algorithm
maxiter = 70 # number of OSEM iterations
lam = .1 # regularization parameter

#### Corrections/Normalization
pt1_path = "preprocessing/cumulative_pt1b_corrected.npy"
sensi_path = "preprocessing/310_uniform_rec_tv_butt50.npy"
att_path = None

#### Other
background = 1e-5 # can be array, size of mouse_wp # noise
data = 1. # can be array, size of mouse_wp # for weighting data if applicable



######### RUN
#### LOAD DATA
mouse = np.load(mouse_path)
plaque = np.load(plaque_path)
# Downsample
mouse = mouse[:, ::mouse_down_factor, :]
plaque = plaque[:, ::plaque_down_factor, :]
# Concatenate and shuffle
mouse_wp = np.hstack((mouse, plaque))
rng = numpy.random.default_rng(0)
mouse_wp = mouse_wp[:, np.array(rng.permutation(mouse_wp.shape[1])), :] 

#### DEFINE PROJECTOR FROM DATA (MOUSE + PLAQUE)
from projector import project_3D_subsets
Ps = project_3D_subsets(mouse_wp, nsubsets, vol_shape=vol_shape, backg=background, data=data,
                        radius=radius, half_axial_length=half_axial_length, 
                        voxel_size=voxel_size, img_origin=origin,
                        check_inside=False)

#### LOAD PT1 and NORMALIZATIONS INTO THE PROJECTOR
from normalizations import load_pt1, compute_normalization_from_mumap_and_sensimap
Ps.pt1 = load_pt1(pt1_path, vol_shape, crop=crop)
Ps.norm = compute_normalization_from_mumap_and_sensimap(Ps, att_path, sensi_path, percentile=20, crop=crop)

#### RUN TV ALGO
comp = np.percentile(Ps.pt1, 20).astype(np.float32)
vol = algos.OSEM_TV(Ps, init=None, maxiter=maxiter, maxiter_prox=30, lam=lam, comp=comp, force_full=False)
vol = algos.OSEM_TV(Ps, init=vol, maxiter=2, maxiter_prox=30, lam=lam, comp=comp, force_full=True) # run two iterations with all dataset to "ensure" convergence

#### PLOT
from plt import plt
pl = plt.imshow(vol.reshape(Ps.vol_shp).sum(axis=1)); plt.colorbar(pl); plt.show()
plt.imshow(vol.reshape(Ps.vol_shp).sum(axis=0)); plt.show()
plt.imshow(vol.reshape(Ps.vol_shp).sum(axis=2)); plt.show()
plt.imshow(vol.reshape(Ps.vol_shp).sum(axis=1)[2:-2, 2:-2]); plt.show()
plt.imshow(vol.reshape(Ps.vol_shp).sum(axis=0)[2:-2, 2:-2]); plt.show()
plt.imshow(vol.reshape(Ps.vol_shp).sum(axis=2)[2:-2, 2:-2]); plt.show()
plt.imshow(vol.reshape(Ps.vol_shp)[...,25:35].sum(axis=2)[2:-2,2:-2]); plt.show()
#plt.imshow(vol.reshape(Ps.vol_shp)[...,35:40].sum(axis=2)[15:70,15:60]); plt.show()

#### TEST
# pl = plt.imshow((vol-np.load("mp_reconstruction.npy")).reshape(Ps.vol_shp).sum(axis=1), vmax=np.max(vol.reshape(Ps.vol_shp).sum(axis=1))); plt.colorbar(pl); plt.show()