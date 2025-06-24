from argeoPET import array_lib as np
# Filtered backprojection but reversed so as to be able to use ray tracing
 
# Adjusted FBP with non-visible angles
def bpf_3D(detections, Ns, plot=False, factor=2., low=1.3, radius=17., tail=22.05, weights=None, is_3d=True, isotropic = False):#(detections, factor = 4, low = 2):
    # factor = 5
    # smooth = 2
    import parallelproj
    
    # Number of voxels
    Ns = int(Ns)
    if is_3d:
        if isotropic:
            vol_shp = tuple(3*[Ns]) #22.05*2/voxel_size
        else:
            vol_shp = (int(Ns*tail/radius), Ns, Ns)
    else:  
        vol_shp = (1, Ns, Ns)
    
    
    # Origin and voxels in physical units
    voxel_size = np.array(3*[2*radius/vol_shp[-1]], dtype=np.float32)
    img_origin = ((-np.array(vol_shp)/ 2 + 0.5) * voxel_size).astype(np.float32)
    
    # LOR coordinates in physical units
    if is_3d:
        xstart = detections[0,...].astype(np.float32)
        xend = detections[1,...].astype(np.float32)
    else:
        dets = np.zeros(tuple((np.array(detections.shape)+np.array([0,0,1])).get()))
        dets[...,1:3] = detections
        xstart = dets[0,...].astype(np.float32)
        xend = dets[1,...].astype(np.float32)


    # setup a "sinogram" full of ones
    sino = np.ones(xstart.shape[0], dtype=np.float32) if weights is None else weights.astype(np.float32)
    
    # allocate memory for the back projection array
    back_img = np.zeros(vol_shp, dtype=np.float32)

    parallelproj.joseph3d_back(xstart, xend, back_img, img_origin, voxel_size, sino, num_chunks=1)
    #plt.imshow(back_img[0,...]); plt.show()
    
    #check radon thesis and two references therein
    if is_3d:
        if isotropic:
            padded_vol_shp = tuple(3*[512])
        else:
            padded_vol_shp = tuple((584,450,450)) #tuple(3*[512])
            
        rec = np.array(back_img)
        rec_f = np.fft.fftn(rec, padded_vol_shp)
        #rec_f = np.fft.fftn(rec, tuple(3*[512]))
        fz_1D = np.fft.fftfreq(rec_f.shape[0])
        fy_1D = np.fft.fftfreq(rec_f.shape[1])
        fx_1D = np.fft.fftfreq(rec_f.shape[2])
        fz, fy, fx = np.meshgrid(fz_1D, fy_1D, fx_1D, indexing="ij")
        limit_angle = np.arctan(22.05/17)#np.pi/4 #should be 22.05 # was 20
        
        F_filter = np.empty_like(rec_f)
        freq_norm = np.sqrt(fx**2+fy**2+fz**2)
        freq_norm_plane = np.sqrt(fx**2+fy**2)
        sin_limit_angle = np.sin(limit_angle)
        freq_mask = freq_norm_plane < freq_norm*sin_limit_angle

        F_filter = freq_norm / (4 * np.arcsin(freq_norm*sin_limit_angle/freq_norm_plane) )
        F_filter[freq_mask] = freq_norm[freq_mask]/(2*np.pi)
        F_filter[0,0,0] = 0

        max_freq = np.max(fx_1D)/factor #2.5
        Window = (0.5 + 0.5*np.cos(np.pi*freq_norm/max_freq))*(freq_norm<max_freq)
        filtered_rec_f = (np.fft.ifftn(rec_f*F_filter*Window)).real 
        #result_fbp = filtered_rec_f[10:rec.shape[0]-10,10:rec.shape[1]-10,10:rec.shape[2]-10]
        result_fbp = filtered_rec_f[:rec.shape[0],:rec.shape[1],:rec.shape[2]]
        
    else:
        closest2 = max(64, int(2 ** np.ceil(np.log2(2 * Ns))))
        
        rec = np.copy(back_img[0,...])
        rec_f = np.fft.fftn(rec, tuple(2*[closest2])) #power of 2
        fy_1D = np.fft.fftfreq(rec_f.shape[0])
        fx_1D = np.fft.fftfreq(rec_f.shape[1])
        fy, fx = np.meshgrid(fy_1D, fx_1D)
        
        F_filter = np.sqrt(fx**2+fy**2)
        max_freq = np.max(fx_1D)/(int(factor/low))#/2.5
        Window = (0.5 + 0.5*np.cos(np.pi*F_filter/max_freq))*(F_filter<max_freq)
        filtered_rec_f = (np.fft.ifftn(rec_f*F_filter*Window)).real
        result_fbp = filtered_rec_f[:rec.shape[0],:rec.shape[0]]

    return result_fbp
# Function that does the whole BPF
#filtered_rec_f = bpf_3D(dets, None)
#plt.imshow((filtered_rec_f).sum(axis=0))