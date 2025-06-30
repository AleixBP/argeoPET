from argeoPET import array_lib as np
# Filtered backprojection but reversed so as to be able to use ray tracing
 
# Adjusted FBP with non-visible angles
def bpf_3D(detections, factor=2., vol_shape=(402,310,310), radius=17., half_axial_length=22.05, limit_angle=None, padded_vol_shape = tuple((584,450,450)), weights=None, is_3d=True):
    import parallelproj
        
    # In 2D vol_shape should be (1, Nx, Ny)
    # Origin and voxels in physical units
    voxel_size = (2*np.array([half_axial_length, radius, radius])/np.array(vol_shape)).astype(np.float32)
    img_origin = ((-np.array(vol_shape)/ 2 + 0.5) * voxel_size).astype(np.float32)
    
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
    back_img = np.zeros(vol_shape, dtype=np.float32)

    # do the backprojection
    parallelproj.joseph3d_back(xstart, xend, back_img, img_origin, voxel_size, sino, num_chunks=1)
    
    if is_3d:
        if padded_vol_shape is None: np.maximum(64, (2 ** np.ceil(np.log2(2 * vol_shape))).astype(int)) # size might explode
            
        rec = np.array(back_img)
        rec_f = np.fft.fftn(rec, padded_vol_shape)
        
        fz_1D = np.fft.fftfreq(rec_f.shape[0])
        fy_1D = np.fft.fftfreq(rec_f.shape[1])
        fx_1D = np.fft.fftfreq(rec_f.shape[2])
        fz, fy, fx = np.meshgrid(fz_1D, fy_1D, fx_1D, indexing="ij")
        if limit_angle is None: limit_angle = np.arctan(half_axial_length/radius)
        
        F_filter = np.empty_like(rec_f)
        freq_norm = np.sqrt(fx**2+fy**2+fz**2)
        freq_norm_plane = np.sqrt(fx**2+fy**2)
        sin_limit_angle = np.sin(limit_angle)
        freq_mask = freq_norm_plane < freq_norm*sin_limit_angle

        F_filter = freq_norm / (4 * np.arcsin(freq_norm*sin_limit_angle/freq_norm_plane) )
        F_filter[freq_mask] = freq_norm[freq_mask]/(2*np.pi)
        F_filter[0,0,0] = 0

        max_freq = np.max(fx_1D)/factor
        Window = (0.5 + 0.5*np.cos(np.pi*freq_norm/max_freq))*(freq_norm<max_freq)
        filtered_rec_f = (np.fft.ifftn(rec_f*F_filter*Window)).real 
        #result_fbp = filtered_rec_f[10:rec.shape[0]-10,10:rec.shape[1]-10,10:rec.shape[2]-10]
        result_fbp = filtered_rec_f[:rec.shape[0],:rec.shape[1],:rec.shape[2]]
        
    else:
        padded_vol_shape = np.maximum(64, (2 ** np.ceil(np.log2(2 * vol_shape))).astype(int)) # closest power of 2
        
        rec = np.copy(back_img[0,...])
        rec_f = np.fft.fftn(rec, padded_vol_shape) 
        fy_1D = np.fft.fftfreq(rec_f.shape[0])
        fx_1D = np.fft.fftfreq(rec_f.shape[1])
        fy, fx = np.meshgrid(fy_1D, fx_1D)
        
        F_filter = np.sqrt(fx**2+fy**2)
        max_freq = np.max(fx_1D)/factor
        Window = (0.5 + 0.5*np.cos(np.pi*F_filter/max_freq))*(F_filter<max_freq)
        filtered_rec_f = (np.fft.ifftn(rec_f*F_filter*Window)).real
        result_fbp = filtered_rec_f[:rec.shape[0],:rec.shape[0]]

    return result_fbp
# Function that does the whole BPF
#filtered_rec_f = bpf_3D(dets, None)
#plt.imshow((filtered_rec_f).sum(axis=0))