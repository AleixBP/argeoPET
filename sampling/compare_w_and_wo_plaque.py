import sys
sys.path.append('../')
sys.path.append('../../')
from argeoPET import array_lib as np
import numpy
import argeoPET.optimization_algorithms as algos
#np.cuda.Device(3).use()
#EVERYTHING that goes into the projector MUST BE FLOAT32


######### INPUT
#### DATA
mouse_path = "../data/mouse_background.npy"; mouse_down_factor = int(1) #8
plaque_path = "../data/plaque.npy"; plaque_down_factor = int(1) #16    

#### SCANNER FOV size (in same units as data)
radius = 17.; half_axial_length = 22.05 
# Creates an extruded square for the reconstruction where a cylinder of radius "radius" and length "half_axial_length" is inscribed
# Data points are taken wrt to this. origins is at -r and -half_axial_length.

#### Reconstruction parameters
Ns = 310; vol_shape = (int(Ns*half_axial_length/radius), Ns, Ns) #(1, Ns, Ns) for 2D
nsubsets = 30 # use 1/30th of the data at a time for the OSEM algorithm
maxiter = 70 # number of OSEM iterations
lam = 1. # regularization parameter

#### Corrections/Normalization
pt1_path = "../preprocessing/cumulative_pt1b_corrected.npy"
sensi_path = "../preprocessing/310_uniform_rec_tv_butt50.npy"
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
from argeoPET.projector import project_3D_subsets
Ps = project_3D_subsets(mouse_wp, nsubsets, vol_shape=vol_shape, backg=background, data=data,
                        radius=radius, half_axial_length=half_axial_length, check_inside=False)

#### LOAD PT1 and NORMALIZATIONS INTO THE PROJECTOR
from argeoPET.normalizations import load_pt1, compute_normalization_from_mumap_and_sensimap
Ps.pt1 = load_pt1(pt1_path, vol_shape)
Ps.norm = compute_normalization_from_mumap_and_sensimap(Ps, att_path, sensi_path, percentile=20)

#### RUN TV ALGO
comp = np.percentile(Ps.pt1, 20).astype(np.float32)
vol = algos.OSEM_TV(Ps, init=None, maxiter=maxiter, maxiter_prox=30, lam=lam, comp=comp, force_full=False)
vol = algos.OSEM_TV(Ps, init=vol, maxiter=2, maxiter_prox=30, lam=lam, comp=comp, force_full=True) # run two iterations with all dataset to "ensure" convergence

#### PLOT
from plt import plt
pl = plt.imshow(vol.reshape(Ps.vol_shp).sum(axis=1)); plt.colorbar(pl)
plt.imshow(vol.reshape(Ps.vol_shp).sum(axis=0))
plt.imshow(vol.reshape(Ps.vol_shp).sum(axis=2))
plt.imshow(vol.reshape(Ps.vol_shp).sum(axis=1)[20:390, 20:280])
plt.imshow(vol.reshape(Ps.vol_shp).sum(axis=0)[20:280, 20:280])
plt.imshow(vol.reshape(Ps.vol_shp).sum(axis=2)[50:350, 50:250])

# CREATE LIKELIHOOD
import argeoPET.functional_classes as functional_classes
k = functional_classes.KLP(Ps)
prior = functional_classes.GradientBasedPrior(functional_classes.GradientOperator(), functional_classes.GradientNorm(name="l2_l1"), lam)
cost = k(vol, s=np.s_[:]) + prior(vol.reshape(Ps.vol_shp))


# MOUSE WITHOUT PLAQUE
Ps_noplaque = project_3D_subsets(mouse, nsubsets, vol_shape=vol_shape, backg=background, data=data,
                        radius=radius, half_axial_length=half_axial_length, check_inside=False)
Ps_noplaque.pt1 = load_pt1(pt1_path, vol_shape)
Ps_noplaque.sensi = compute_normalization_from_mumap_and_sensimap(Ps_noplaque, att_path, sensi_path, percentile=20)

del mouse; del plaque; del mouse_wp; del background; del data

# RUN TV ALGO
comp = np.percentile(Ps_noplaque.pt1, 20).astype(np.float32)
vol_noplaque = algos.OSEM_TV(Ps_noplaque, init=None, maxiter=70, maxiter_prox=30, lam=lam, comp=comp)
vol_noplaque = algos.OSEM_TV(Ps_noplaque, init=vol_noplaque, maxiter=2, maxiter_prox=30, lam=lam, comp=comp, force_full=True)

# PLOT
pl = plt.imshow(vol_noplaque.reshape(Ps.vol_shp).sum(axis=1), vmax=np.max(vol.reshape(Ps.vol_shp).sum(axis=1))); plt.colorbar(pl)
plt.imshow(vol_noplaque.reshape(Ps.vol_shp).sum(axis=0))
plt.imshow(vol_noplaque.reshape(Ps.vol_shp).sum(axis=2))
plt.imshow(vol_noplaque.reshape(Ps.vol_shp).sum(axis=2)[50:350, 50:250])

# WILL USE MOUSE+PLAQUE LIKELIHOOD, NOT THIS ONE
#k_noplaque = functional_classes.KLP(Ps_noplaque)
#prior_noplaque = functional_classes.GradientBasedPrior(functional_classes.GradientOperator(), functional_classes.GradientNorm(name="l2_l1"), lam)
#cost_noplaque = k(vol_noplaque, s=np.s_[:]) + prior_noplaque(vol_noplaque.reshape(Ps.vol_shp))


# COST WITH NO PLAQUE
cost_noplaque = k(vol_noplaque, s=np.s_[:]) + prior(vol_noplaque.reshape(Ps.vol_shp))
print(cost_noplaque)

# COST WITH PLAQUE
print(cost)

# Cost with plaque (MAP) + bound
alpha = 0.05
cb = cost + Ps.vol_dim + Ps.vol_dim*np.sqrt(16*np.log(3/alpha)/Ps.vol_dim)
print(cb, cost + Ps.vol_dim, Ps.vol_dim*np.sqrt(16*np.log(3/alpha)/Ps.vol_dim))
print(cost + Ps.vol_dim*np.sqrt(16*np.log(3/alpha)/Ps.vol_dim))

# Reject that there is no plaque if
bool( cb < cost_noplaque )

# That is, since
print(cb, "is not smaller than", cost_noplaque, "then the one with no plaque is inside the ", alpha, "credible region. Meaning we cannot reject that the plaque is not there.")

# Approximation of the region is linear with Ps.vol_dim in this case it goes too far.
# Estimate is too conservative. Does not reject that there is no plaque.
# This approx looks very bad at this dimension

3*np.exp(-((Ps.vol_dim/16)*(cost_noplaque-cost-Ps.vol_dim)**2))

alpha = 0.000000001

print(cost + Ps.vol_dim*(1+4*np.sqrt(np.log(3/alpha)/Ps.vol_dim)), cost_noplaque)


# Example gradient
grd = k.jacobianT(vol, s=np.s_[:]) #compute gradient at vol

grad_at_vol = k.Ps.pt1 - 1. * k.Ps.adjoint( k.Ps.data[np.s_[:]] / (k.Ps(vol, s=np.s_[:]) + k.Ps.backg[np.s_[:]]), s=np.s_[:])

np.max(np.abs(grd-grad_at_vol))

RL_stepsize = vol/np.maximum(Ps.pt1, comp)
new_vol_grad = vol - grad_at_vol*RL_stepsize
plt.imshow(new_vol_grad.reshape(Ps.vol_shp).sum(axis=0))

grad_at_vol_v2 = np.maximum(k.Ps.pt1, comp) - 1. * k.Ps.adjoint( k.Ps.data[np.s_[:]] / (k.Ps(vol, s=np.s_[:]) + k.Ps.backg[np.s_[:]]), s=np.s_[:])
new_vol_grad_v2 = vol - grad_at_vol_v2*RL_stepsize
plt.imshow(new_vol_grad_v2.reshape(Ps.vol_shp).sum(axis=0))


osem_step = Ps.adjoint(Ps.data(np.s_[:]) / (Ps(vol, s=np.s_[:]) + Ps.backg(np.s_[:])), s=np.s_[:]) / np.maximum(Ps.pt1, comp)
new_vol = osem_step*vol
plt.imshow(new_vol.reshape(Ps.vol_shp).sum(axis=0))

np.max(np.abs(new_vol_grad-new_vol))
np.max(np.abs(new_vol_grad_v2-new_vol))