from argeoPET import array_lib as np

# PROJECTION and BACKPROJECTION class for LM data
class project_3D_subsets():
      
    def __init__(self, detections, nsubsets, norm=None, backg=1e-5, data=1., pt1=None, vol_shape=(402,310,310), radius=17, half_axial_length=22.05, is_3d = True, isotropic = False, num_chunks=1, check_inside=False):  
        
        
        self.vol_shp = vol_shape #2D format is (1, N, N)


        # Origin and voxels in physical units
        self.voxel_size = (2*np.array([half_axial_length, radius, radius])/np.array(vol_shape)).astype(np.float32)
        self.img_origin = ((-np.array(self.vol_shp)/ 2 + 0.5) * self.voxel_size).astype(np.float32)
        self.radius = radius
        self.half_axial_length = half_axial_length
        self.num_chunks = num_chunks
        
        
        # LOR coordinates in physical units
        if is_3d:
            self.xstart = detections[0,...].astype(np.float32)
            self.xend = detections[1,...].astype(np.float32)
        else:
            dets = np.zeros(tuple((np.array(detections.shape)+np.array([0,0,1])).get()))
            dets[...,1:3] = detections
            self.xstart = dets[0,...].astype(np.float32)
            self.xend = dets[1,...].astype(np.float32)
            
        self.sino_shp = self.xstart.shape[0]
        
        
        # Load Projector library
        import parallelproj
        self.parallelproj = parallelproj
        self.ind = 0
        
        
        # Clean rays and recalculate dimensions (changes xstart, xend and sino_shp)
        if check_inside is not False: self.clean_rays(check_inside)
        
        
        # Lets make the subsets
        self.nsubsets = nsubsets
        self.ndets = self.sino_shp
        if False:
            # subsets of the same size, but discard some
            # keeping a full array of indices instead of slices
            self.subsets = np.random.choice(range(self.ndets), size=(nsubsets, self.ndets//nsubsets), replace=False)
        else:
            # split along, no need to store indices, must shuffle xstart/xend first though, 
            # also indexing less flexible in call and adjoint because you cannot send ind=[1,3] nor does s.ndim or ravel(), fuck cannot subclass slice ;(
            #ss = self.ndets//nsubsets
            #self.subsets = [np.s_[0+k*ss:(k+1)*ss] for k in range(nsubsets-1)] + [np.s_[(nsubsets-1)*ss:self.ndets]]
            self.subsets = [np.s_[k::nsubsets] for k in range(nsubsets)]
            self.subsets_sizes = (self.ndets%nsubsets)*[self.ndets//nsubsets+1] + (self.ndets-self.ndets%nsubsets)*[self.ndets//nsubsets]
        
        
        # Sensitivity image
        self.pt1 = pt1
        
        
        # Sinogram sensitivity
        self.norm = iter_ct(a=1.) if norm is None else iter_arr(norm, self.subsets) #else array the size of the number of detection pairs
        
        
        # Background
        self.backg = iter_ct(a=backg) if type(backg) is float else iter_arr(backg, self.subsets)
        

        # Data
        self.data = iter_ct(a=data) if type(data) is float else iter_arr(data, self.subsets)
            
        
        # Compatibility with operators
        self.lip = np.infty
        self.eigen_min = 0
        self.dim = [int(np.prod(np.asarray(self.vol_shp))), int(np.prod(np.asarray(self.sino_shp)))]
        self.vol_dim = self.dim[0]
        self.sino_dim = self.dim[1]
        
        
        
    def __call__(self, vol, ind=None, s=None):
        # the input is an image vol of size np.ones(self.vol_shp, dtype=np.float32)
        
        # automatic random
        ind = int(np.random.randint(0, self.nsubsets)) if ind is None else ind
        s = self.subsets[ind] if s is None else s #self.subsets[ind].ravel() 
        
        # allocate memory for the forward projection array, force slice count via xstart
        xs = self.xstart[s]
        sino = np.zeros(len(xs), dtype=np.float32)
        
        # project
        self.parallelproj.joseph3d_fwd(xs, self.xend[s],
                                       vol.reshape(self.vol_shp),
                                       self.img_origin, self.voxel_size, 
                                       sino,
                                       num_chunks=self.num_chunks)
        
        # compensate sensitivity
        return (self.norm[s] * sino).ravel()
    
    
    def adjoint(self, sino, ind=None, s=None):
        # the input is a "sinogram" sino of shape np.ones(xstart.shape[0], dtype=np.float32)
        
        # grab random
        ind = self.ind if ind is None else ind
        s = self.subsets[ind] if s is None else s #self.subsets[ind].ravel() 

        # allocate memory for the back projection array
        vol = np.zeros(self.vol_shp, dtype=np.float32)
        
        # project
        self.parallelproj.joseph3d_back(self.xstart[s], self.xend[s],
                                        vol,
                                        self.img_origin, self.voxel_size,
                                        self.norm[s] * sino,
                                        num_chunks=self.num_chunks)
        
        return vol.ravel()
        
        
    def clean_rays(self, percent=0.):
    
        #px = self.__call__(np.ones(self.vol_shp, dtype=np.float32))
        sino = np.zeros(self.xstart.shape[0], dtype=np.float32) #allocate
        self.parallelproj.joseph3d_fwd(self.xstart, self.xend, 
                                       np.ones(self.vol_shp, dtype=np.float32), #send ones
                                       self.img_origin, self.voxel_size, 
                                       sino, 
                                       num_chunks=self.num_chunks)
        
        mask = (sino>self.voxel_size[-1]*self.vol_shp[-1]*percent) #check rays
        self.xstart = self.xstart[mask,:]
        self.xend = self.xend[mask,:]
        
        self.sino_shp = self.xstart.shape[0]
        #self.dim = [int(np.prod(np.asarray(self.vol_shp))), int(np.prod(np.asarray(self.sino_shp)))]
 
    
    def apply_all_adjoint(self, y, cumulative=False):
        
        if not cumulative:
            x = self.adjoint(y, s=np.s_[:])
        
        else:
            x = np.zeros(self.vol_dim, dtype=np.float32)
            for subset in self.subsets:
                x += self.adjoint(y[subset], s=subset)
                
        return x
        
        
    def create_norm_adjoints(self):
        self.Ps_na = []
        for ind in range(self.nsubsets):
            self.Ps_na.append(self.adjoint(np.ones(self.subsets_sizes[ind], dtype=np.float32), s=self.subsets[ind]))
        
        
    def run_lips_svd(self, all=True):
        import cupyx.scipy.sparse.linalg as sclin
        self.lips = []
        for subset in self.subsets:
            matv = lambda x: self.__call__(x, s=subset)
            rmatv = lambda x: self.adjoint(x, s=subset)
            A = sclin.LinearOperator((subset.ndim, self.vol_dim), matvec=matv, rmatvec=rmatv, dtype=np.float32)
            maxi_svd = float(sclin.svds(A, k=1, tol=1e-1, maxiter=5, return_singular_vectors=False))
            
            if not all:
                lips = self.nsubsets*[maxi_svd]
                break
            
            lips.append(maxi_svd)
        
        return self.lips
    
    
    def run_lips(self): # i think it does not make sense
        
        lips = []
        for subset in self.subsets:
            x_prev = np.ones(self.vol_dim, dtype=np.float32)
            for k in range(10):
                x = self.adjoint(self.__call__(x_prev, s=subset), s=subset)
                x[x==0] = 1.
                r = x/x_prev
                lb = np.min(r)
                ub = np.max(r)
                x_prev = 2*x/(lb + ub)
                
                if ub/lb < 1.: break
            lips.append(ub)
            
        return lips
    
    
    def test_total_subset_size_vs_dets(self):
        a = 0
        for i in range(self.nsubsets):
            a += len(self.xstart[self.subsets[i]])
        return (a, len(self.xstart))
        
        
    def test_adjoint(self, atol = 1e-8, rtol = 1e-5):
        #Adjoint test
        rvol = np.maximum(np.random.rand(*self.vol_shp, dtype=np.float32).ravel(), 0)
        rsino = np.maximum(np.random.rand(self.sino_shp, dtype=np.float32).ravel(), 0)
        a1 = np.dot(self.__call__(rvol, s=np.s_[:]), rsino)
        a2 = np.dot(self.adjoint(rsino, s=np.s_[:]), rvol)
        print(np.abs(a1 - a2) <= (atol + rtol * np.abs(a1)))
        return a1-a2


class Subset_Proj_Op():
    def __init__(self, Ps, k=None, s=None):
        self.Ps = Ps
        self.k = k
        self.s = s
        self.nsubsets = 1
        # carry over shapes too?
        
    def __call__(self, x, **kwargs):
        return self.Ps(x, ind=self.k, s=self.s)
        
    def adjoint(self, x, **kwargs):
        return self.Ps.adjoint(x, ind=self.k, s=self.s)


class iter_ct:
    def  __init__(self, a=1.):
        self.a = a
    def __getitem__(self, i):
        return self.a
    def __call__(self, i):
        return self.a
    

class iter_arr:
    def  __init__(self, arr, subs):
        self.arr = arr
        self.subs = subs
    def __getitem__(self, s):
        return self.arr[s]    
    def __call__(self, ind=None, s=None):
        s = self.subs[ind] if s is None else s
        return self.arr[s]
