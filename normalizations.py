import sys
sys.path.append('../')
from argeoPET import array_lib as np


def compute_sensitivity_from_sensimap(Ps, sensi_path=None, new_volume_shape = None, yes_mask=False, direct_domain=True):
    
    if sensi_path is None:
        
        return Ps.sensi
    
    
    if direct_domain: # from reconstructed volume
        
        sensi_vol = 1000.*np.load(sensi_path)
        
        if new_volume_shape is not None: # this should be equal to Ps.vol_shp
            if "numpy" in np.__name__: from scipy.ndimage import zoom
            if "cupy" in np.__name__: from cupyx.scipy.ndimage import zoom
            sensi_vol = np.array(zoom(sensi_vol, (np.array(new_volume_shape)/np.array(sensi_vol.shape)) )).astype(np.float32)
            sensi_vol = sensi_vol.flatten().astype(np.float32)
        
        sensi = Ps(sensi_vol, s=np.s_[:]).astype(np.float32)

        if yes_mask:
            mask = sensi>0.
            sensi = sensi[mask].astype(np.float32)
            return sensi, mask
        
        else:
            return sensi

    else: #from sinogram histogram
        
        from geometry import angle_distance_3D
        pre_sino_data = np.array( angle_distance_3D(Ps.xstart, Ps.xend) ).T
        
        hs = np.load(sensi_path)
        hist_values, sp = hs["arr_0"], hs["arr_1"]

        sp_diff = sp[:,1]-sp[:,0]
        sp_off = np.min(sp, axis=1)
        indexes = (( (pre_sino_data-sp_off) // sp_diff).astype(np.uint16) + np.array([0,0,0,0], dtype=np.uint16)).T #uint16 is (0 to 65_535) but only need 0 to 40 or to 80
        indexes = tuple(np.vsplit(indexes, indexes.shape[0]))
        data_weights = hist_values[indexes].squeeze()

        if yes_mask:
            mask = data_weights>0.
            sensi = data_weights[mask].astype(np.float32)
            return sensi, mask
        
        else:
            return data_weights.astype(np.float32)

    
def compute_attenuation_from_mumap(Ps, att_path=None, yes_mask=False):
    
    vol_shp = Ps.vol_shp
    
    if att_path is not None: 
        
        att_phantom = np.load(att_path)
        
    else: # Assume a water cylinder inside the rconstruction volume shape
        
        att_phantom = cylinder_mask(vol_shp)
        #https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html
        coeff_water = 0.0097#*Ns/34. #0.097 cm^-1, diameter is 34mm
        att_phantom *= coeff_water
        
    if yes_mask:
        pixel_thresh = 1.*coeff_water # eliminate rays that cross less than 2 pixels
        att_sino = Ps(att_phantom, s=np.s_[:]).astype(np.float64)
        non_mask = att_sino>pixel_thresh
        att_sino = np.exp(-att_sino[non_mask]).astype(np.float32)
        
        return att_sino, non_mask
    
    else:
        
        return np.exp(-Ps(att_phantom, s=np.s_[:]).astype(np.float64)).astype(np.float32)


def cylinder_mask(vol_shp): #cylinder inscribed in vol_shp (considering square x,y)
    
    center = np.array(vol_shp[1:])/2.
    phantom_radius = center[0]#*33.5/34.
    circle = np.linalg.norm(np.mgrid[:vol_shp[1], :vol_shp[2]]-center[:,np.newaxis,np.newaxis], \
                               ord=2, axis=0)<phantom_radius
    
    return np.repeat(circle[np.newaxis,:], repeats=vol_shp[0],  axis=0).astype(np.float32)


def compute_normalization_from_mumap_and_sensimap(Ps, att_path=None, sensi_path=None, percentile=20):
    
    att = compute_attenuation_from_mumap(Ps, att_path)
    sensi = compute_sensitivity_from_sensimap(Ps, sensi_path)
    sensi = np.maximum(sensi, np.percentile(sensi, percentile)).astype(np.float32)
    
    return att*sensi


def load_pt1(pt1_path = None, new_volume_shape = None, Ps=None):
    
    if pt1_path is not None: 
        
        pt1 = np.load(pt1_path)
        
        if new_volume_shape is not None: # if pt1 was computed for another volume shape
                if "numpy" in np.__name__: from scipy.ndimage import zoom
                if "cupy" in np.__name__: from cupyx.scipy.ndimage import zoom
                pt1 = np.array(zoom(pt1, (np.array(new_volume_shape)/np.array(pt1.shape)) )).astype(np.float32)
                
        pt1 = pt1.flatten().astype(np.float32)
        
        
    else: # very broad approximation, not recommended
        
        Ps.pt1 = Ps.apply_all_adjoint(np.ones(Ps.sino_shp, dtype=np.float32))
        
    
    return pt1
    
    
#hist_values = np.load("/home/boquet/muPET/geant/sampling/uniform_hist_values_4D.npy").astype(np.float32)
#sp = np.load("/home/boquet/muPET/geant/sampling/uniform_hist_edges_4D.npy")
#np.savez("uniform_hist_values_edges_4D.npz", hist_values.get(), sp.get())

#"/home/boquet/muPET/geant/sampling/310_uniform_rec_tv_butt50.npy"

#PT1 and the sensitivity map are adjusted with zoom to match the new volume shape as they need to be run at a certain shape themselves