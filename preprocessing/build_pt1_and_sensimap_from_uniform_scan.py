import sys
sys.path.append('../')
sys.path.append('../../')
from argeoPET import array_lib as np
from geant4_data_loaders import data_loader
from plt import plt
import glob
np.cuda.Device(3).use()

# Regard this file as a jupyter-notebook
########
# Build PT1 volume and Sensitivity volume from a long scan of a uniform cylinder (a lot of data files, 120G)
# Both are unique to each scanner and are needed for a proper reconstruction  
########

########################
########################
########################
########################
######## DIRECT DOMAIN
#LOAD DATA
file_names = glob.glob("/home/boquet/data/uniform/*.npy")
file_names += glob.glob("/home/saidi/202401_SensitivityCorrection/*.npy")

volume_shape = (402, 310, 310)

########################
######## COMPUTE PT1
# Do it iteratively given that the backprojection (PT) is linear
plot = False
cumulative_saving = False
pt1_saving_file_path = "results/cumulative_pt1.npy"
nsets = 3 #this means 1/3rd of total files at once
nsets_used = 3 #this means how many of these 1/3rds the algo will go trough (cumulatively)
nsubsets = 8 # the projectors uses 1/8th of the current data at a time
cumulative_pt1 = np.zeros(volume_shape)
correctfor_length = True; len_thres=0.4 #in mm
correctfor_attenuation = False
impose_attenuation = 0.0097 #in mm^-1 otherwise None

if cumulative_saving:
    np.save(pt1_saving_file_path, cumulative_pt1)

# Note that the dets variable is reinitialized for every k, so only for cumulative stuff
for k in range(nsets_used):
    
    # SELECT SUBSET
    subset_file_names = file_names[k::nsets]
    dets = data_loader(subset_file_names[0])
    
    # LOAD DATA
    for file_name in subset_file_names[1:]:
        dets1 = data_loader(file_name)
        dets = np.hstack((dets, dets1))
    del dets1


    #### PRE-COMPUTED data
    pt1 = None # size of image/volume
    sensi = None # can be array, size of mouse_wp
    backg = 1e-5 # can be array, size of mouse_wp
    data = 1. # can be array, size of mouse_wp


    #### PROJECTOR
    from projector import project_3D_subsets
    
    
    circle_inscribed = False
    sq2 = float(np.sqrt(2)) if circle_inscribed else 1.
    check_inside = False if circle_inscribed else 0.01
    radius = 17./sq2; half_axial_length = 22.05/sq2

    if correctfor_length or correctfor_attenuation or impose_attenuation: check_inside = False #already cleaning after

    Ps = project_3D_subsets(dets, nsubsets, vol_shape=volume_shape, sensi=sensi, backg=backg, data=data, pt1=pt1,
                            radius=radius, half_axial_length=half_axial_length, check_inside=check_inside)
        
    #### Correction of integration of ray length and attenuation
    if correctfor_length or correctfor_attenuation or impose_attenuation:
        
        from normalizations import cylinder_mask
        phantom = cylinder_mask(Ps.vol_shp)
        ray_length = Ps(phantom,  s=np.s_[:])
        non_mask = (ray_length>len_thres)
        
        dets = dets[:,non_mask,:]
        ray_length = ray_length[non_mask]
        # redefine Ps
        Ps = project_3D_subsets(dets, nsubsets, vol_shape=volume_shape, sensi=sensi, backg=backg, data=data, pt1=pt1,
                            radius=radius, half_axial_length=half_axial_length, check_inside=check_inside)
    
    sino_len_att = np.ones(Ps.sino_shp, dtype=np.float32)
    
    if correctfor_length:
        sino_len_att /= ray_length
        
    if correctfor_attenuation:
        #sino_att = np.exp(-0.0097*ray_length.astype(np.float64)).astype(np.float32) #assuming the calibration is in homogeneous water, otherwise include coeff in the volume (coeff in mm because radius in mm)
        #actually they are in air https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/air.html #0.08712 cm^2/g * 1.225e-3 g/cm^3 so negligible
        sino_len_att /= np.ones(Ps.sino_shp, dtype=np.float32)
        
    if impose_attenuation is not None:
        att_coeff = impose_attenuation
        sino_len_att *= np.exp(-att_coeff*ray_length.astype(np.float64)).astype(np.float32)
    
    #sino_len_att = (1./(sino_len*sino_att)).astype(np.float32)   #=np.ones(Ps.sino_shp, dtype=np.float32)
    del phantom; del non_mask
    #### Correction ends
    
    #### Backproject
    Ps.pt1 = pt1 if pt1 is not None else Ps.apply_all_adjoint(sino_len_att)
    del sino_len_att
    
        
    #### Maybe plot
    if plot:
        plt.imshow(Ps.pt1.reshape(Ps.vol_shp).sum(axis=0))
        plt.imshow(Ps.pt1.reshape(Ps.vol_shp).sum(axis=1))
        plt.imshow(Ps.pt1.reshape(Ps.vol_shp).sum(axis=2))
        print(Ps.pt1.reshape(Ps.vol_shp).sum(axis=0).min(), Ps.pt1.reshape(Ps.vol_shp).sum(axis=0).max())
        print(Ps.pt1.reshape(Ps.vol_shp).sum(axis=1).min(), Ps.pt1.reshape(Ps.vol_shp).sum(axis=1).max())
        print(Ps.pt1.reshape(Ps.vol_shp).sum(axis=2).min(), Ps.pt1.reshape(Ps.vol_shp).sum(axis=2).max())
    
    
    #### Cumulate
    if cumulative_saving:
        old_pt1 = np.load(pt1_saving_file_path)
        np.save(pt1_saving_file_path, Ps.pt1.reshape(Ps.vol_shp) + old_pt1)
        
    else:
        cumulative_pt1 += Ps.pt1.reshape(Ps.vol_shp)
        
if not cumulative_saving:
    np.save(pt1_saving_file_path, cumulative_pt1)
    #np.save("cumulative_pt1b.npy", cumulative_pt1)
    #np.save("cumulative_pt1b_corrected.npy", cumulative_pt1)
    
        
## Explore results by plotting
plt.imshow(cumulative_pt1.sum(axis=0)); plt.colorbar(); plt.show()
plt.imshow(cumulative_pt1.sum(axis=1))
plt.imshow(cumulative_pt1.sum(axis=2))
print(cumulative_pt1.sum(axis=0).min(), cumulative_pt1.sum(axis=0).max())
print(cumulative_pt1.sum(axis=1).min(), cumulative_pt1.sum(axis=1).max())
print(cumulative_pt1.sum(axis=2).min(), cumulative_pt1.sum(axis=2).max())



########################
#### Compute sensitivity by reconstructing the cylinder via MAP (cylinder ideally takes the entire FOV)

vol = None
sensi_saving_file_path = "results/none.npy"

## Three ways of reconstructing # Prefer OSEM_TV
if False:
    ##------------------------------------------------------##
    ### FBP
    from argeoPET.bpf import bpf_3D
    # TO DO: do not really need the data at once here since im filtering after
    #nsets = 4; nsets_used = 1
    limit_angle = np.arctan(half_axial_length/radius)
    vol = bpf_3D(dets, vol_shape=Ps.vol_shp, radius=radius, half_axial_length=half_axial_length, factor=2., limit_angle=limit_angle, padded_vol_shape = tuple((584,450,450))) 
    plt.imshow(vol[10:-10,10:-10,10:-10].sum(axis=0))
    plt.imshow(vol[10:-10,10:-10,10:-10].sum(axis=1))
    plt.imshow(vol[10:-10,10:-10,10:-10].sum(axis=2))

elif False:
    ##------------------------------------------------------##
    ### OSEM 
    # solves KL alone
    if vol is None: vol = np.ones(Ps.vol_dim, dtype=np.float32) #init
    maxiter = 70
    comp = 1e5#np.finfo(vol.dtype).eps*1e4*Ps.vol_dim
    sensi_image = Ps.pt1 + comp
    Ps.create_norm_adjoints()

    for k in range(maxiter):
        ind = int(np.random.randint(0, Ps.nsubsets))
        vol *= Ps.adjoint(Ps.data(ind) / (Ps(vol, ind=ind) + Ps.backg(ind)), ind=ind) / sensi_image#/(Ps.Ps_na[ind]+comp) #sensi_image
        
    plt.imshow(vol.reshape(Ps.vol_shp).sum(axis=1))
    plt.imshow(vol.reshape(Ps.vol_shp).sum(axis=0))
    plt.imshow(vol.reshape(Ps.vol_shp).sum(axis=1))
    plt.imshow(vol.reshape(Ps.vol_shp)[(201-50):(201+50),:,:].sum(axis=0))
    from skimage.filters import butterworth
    result_butt = butterworth(vol.reshape(Ps.vol_shp).get(), cutoff_frequency_ratio=.05, high_pass = False, order=3)
    plt.imshow(result_butt.sum(axis=1))
    plt.imshow(result_butt.sum(axis=0))
    #np.save("310_uniform_rec_osem.npy", np.array(result_butt))

else: #preferred
    
    do_filter = .075# None
    from argeoPET.optimization_algorithms import OSEM_TV
    vol = OSEM_TV(Ps, init=None, maxiter=70, maxiter_prox=30, lam=700.)
    
    if do_filter is None:
        np.save(sensi_saving_file_path, np.array(vol.reshape(Ps.vol_shp)))
        #np.save("310_uniform_rec_tv.npy", np.array(vol.reshape(Ps.vol_shp)))
        plt.imshow(vol.reshape(Ps.vol_shp).sum(axis=0))
        plt.imshow(vol.reshape(Ps.vol_shp).sum(axis=1))
        plt.imshow(vol.reshape(Ps.vol_shp).sum(axis=2))
        
    else:
        from skimage.filters import butterworth
        result_butt = butterworth(vol.reshape(Ps.vol_shp).get(), cutoff_frequency_ratio=do_filter, high_pass = False, order=3)
        np.save(sensi_saving_file_path, np.array(result_butt))
        #np.save("310_uniform_rec_tv_butt.npy", np.array(result_butt))
        plt.imshow(result_butt.sum(axis=0))
        plt.imshow(result_butt.sum(axis=1))
        plt.imshow(result_butt.sum(axis=2))
        
            
            
              









########################
########################
########################
########################
######## SINOGRAM DOMAIN

########################
## Cumulative histogram in 3D (4D param)
## Build sensitivity map in histogram domain
if True:
    import sys
    sys.path.append('../')
    sys.path.append('../../')
    import cupy as np
    from geant4_data_loaders import data_loader
    from plt import plt
    import glob
    np.cuda.Device(3).use()
    from geometry import angle_distance_3D

    file_names = glob.glob("/home/boquet/data/uniform/*.npy")
    file_names += glob.glob("/home/saidi/202401_SensitivityCorrection/*.npy")

    diag = 1.
    r_max = (17**2+22**2)**(1/2) # biggest diagonal in the scanner
    domain_s2 = diag*np.array([-53.,  53.]) # range for s2
    domain_s3 = diag*np.array([-44.,  44.])
    rg = np.array([np.array([0,np.pi]), np.array([-np.pi/2.,np.pi/2]), domain_s2, domain_s3])

    nsets = 20 #4, before
    nsets_used = 20

    bins = 4*[40] #80 looks fine, but should probably go for 40 #170-180 bins is max for error
    hists = np.zeros(bins)
    for k in range(nsets_used):
        
        # SELECT SUBSET
        subset_file_names = file_names[k::nsets]
        dets = data_loader(subset_file_names[0])
        
        # LOAD DATA
        for file_name in subset_file_names[1:]:
            dets1 = data_loader(file_name)
            dets = np.hstack((dets, dets1))
        print(dets.shape)
        
        sino_data = np.array( angle_distance_3D(*dets) ).T
        del dets; del dets1
        np.max(sino_data, axis=0)
        np.min(sino_data, axis=0)
        hist_values, *sp = np.histogramdd(sino_data, bins=bins, range=rg, weights=None)
    
        hists += hist_values
        del hist_values;
        
    plt.imshow(hists[:,:,int(bins[2]/2),int(bins[3]/2)]); plt.show()
    plt.imshow(hists[:,int(bins[1]/2),:,int(bins[3]/2)]); plt.show()#s2 limited to radius here
    plt.imshow(hists[int(bins[0]/2),:,:,int(bins[3]/2)]); plt.show()
    np.max(hists)
    np.argmax(hists)

    #np.save("uniform_hist_values_4D_40.npy", hists) #/home/boquet/muPET/geant/sampling
    #np.save("uniform_hist_edges_4D_40.npy", np.array(sp[0]))
    #np.savez("uniform_hist_values_edges_4D_40.npz", hists, np.array(sp[0]))
    


########################
## Cumulative histogram in 2D (2D param)
## Build sensitivity map in histogram domain
if True:
    import cupy as np
    from geant4_data_loaders import data_loader
    from plt import plt
    import glob
    np.cuda.Device(3).use()

    file_names = glob.glob("/home/boquet/data/uniform/*.npy")
    file_names += glob.glob("/home/saidi/202401_SensitivityCorrection/*.npy")

    from muPET.resampling.resample import dens_resample, hist_dens_estim, kde_dens_estim
    diag = 1.
    domain_s = diag*np.array([-17.,  17.])
    rg = np.array([np.array([0,np.pi]), domain_s])

    nsets = 10 #4, before
    nsets_used = 10

    bins = (200,200)
    hists = np.zeros(bins)
    for k in range(nsets_used):
        
        # SELECT SUBSET
        subset_file_names = file_names[k::nsets]
        dets = data_loader(subset_file_names[0])
        
        # LOAD DATA
        for file_name in subset_file_names[1:]:
            dets1 = data_loader(file_name)
            dets = np.hstack((dets, dets1))
        print(dets.shape)
        
        detections = filter_z(dets, th=22.1) #10, 100
        detections = project_z(detections)
        phi, s = angle_distance(*detections)
        pre_sino_data = np.stack((phi,s), axis=1)
        sino_data = filter_s(pre_sino_data, th=17.)
            
        hist_values, *sp = np.histogram2d(*sino_data.T, bins=bins, range=rg, weights=None)
        hists += hist_values
        
        del dets;
        
        plt.imshow(hists.T); plt.show()
        
    plt.imshow(hists.T); plt.show()
    np.max(hists)
    np.argmax(hists)

    #np.save("uniform_hist_values.npy", hists) #/home/boquet/muPET/geant/sampling
    #np.save("uniform_hist_edges.npy", np.array(sp))



########################
## Explore histogram
if True: 
    import sys
    sys.path.append('../')
    sys.path.append('../../')
    from muPET.simulations.simulator import angle_distance, image_linear_spline_2D
    def filter_z(detections, th=5., plot=False): 
        if plot: plt.hist((detections.reshape(-1,3))[:,0]); plt.show()

        # check all elements below threhold
        either_smaller = np.abs(detections[:,:,0])<th

        # check that both pairs are True
        both_smaller = np.sum(either_smaller, axis=0)>1 

        return detections[:, both_smaller, :]

    def project_z(detections):
        return detections[:,:,[1,2]] 

    # cut sinogram according to field of view
    def filter_s(sino, th=18.575):
        s_smaller = np.abs(sino[:,1]) < th
        return sino[s_smaller,:]


    np.max(dets[0,:,0])
    np.min(dets[0,:,0])

    detections = filter_z(dets, th=22.1) #10, 100
    detections = project_z(detections)
    phi, s = angle_distance(*detections)
    pre_sino_data = np.stack((phi,s), axis=1)
    sino_data = filter_s(pre_sino_data, th=17.)
    phi, s = sino_data.T
    sino = filter_s(sino_data, th=1.)

    del pre_sino_data; del detections; del dets; del phi; del s

    if want_plot:
        plt.hist(s); plt.show()
        pl = plt.hist2d(*sino_data.T, bins=150); plt.show()
        plt.hist(sino[:,0],bins=100); plt.show()
        pl = plt.hist2d(*sino.T, bins=100); plt.show()
    
    
    pl = plt.hist2d(*sino_data.T, bins=(310,310)); plt.show()
    plt.imshow(pl[0].T)

    from muPET.resampling.resample import dens_resample, hist_dens_estim, kde_dens_estim
    diag = 1.
    domain_s = diag*np.array([-17.,  17.])
    rg = np.array([np.array([0,np.pi]), domain_s])
    sino_data

    sampled_s = np.linspace(*domain_s, 100)
    sampled_phi = np.linspace(0, np.pi, 100, endpoint=False)
    huhu = dens_resample(sino_data, sampled_phi, sampled_s, hist_dens_estim, args=[150, rg, None])
    plt.imshow(huhu.T)

    hist_values, *sp = np.histogram2d(*sino_data.T, bins=150, range=rg, weights=None)
    plt.imshow(hist_values.T, aspect="auto")

    pl = plt.hist2d(*sino_data.T, bins=150, range=rg, weights=None); plt.show()


