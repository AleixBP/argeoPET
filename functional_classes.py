from argeoPET import array_lib as np


# KLP that exploits the fact that PT1 is already computed
class KLP():
    # working with -loglikelihood

    def __init__(self, Ps):

        self.Ps = Ps


    def __call__(self, vol, ind=None, s=None, cps=1.):
        #TO DO or TO TAKE INTO ACCOUNT adjust cps with the size of "s"
        
        s = self.Ps.subsets[ind] if s is None else s

        if np.any(vol<0):
            return np.inf
        else:        
            return np.sum(self.Ps.pt1 * vol) \
                   - cps * np.sum(self.Ps.data[s] \
                     * np.log(self.Ps(vol, s=s) + self.Ps.backg[s]))
            

    def jacobianT(self, vol, ind=None, s=None, comp=None, cps=1.):
        
        s = self.Ps.subsets[ind] if s is None else s 
        
        sensi_image = self.Ps.pt1 if comp is None else np.maximum(self.Ps.pt1, comp)
    
        return  sensi_image \
                - cps * self.Ps.adjoint( \
                  self.Ps.data[s] / (self.Ps(vol, s=s) + self.Ps.backg[s]), s=s)
        
        
    def test_gradient(self, ind=None, s=None, start=-8, end=10, plot=True):
        s = self.Ps.subsets[ind] if s is None else s
        
        vol = np.random.rand(self.Ps.vol_dim, dtype=np.float32)
        epses = [10**k for k in range(start, end)]
        vol_direction = np.random.rand(self.Ps.vol_dim, dtype=np.float32)
        vol_direction /= np.linalg.norm(vol_direction, 2)
        
        resi = []
        resi_rel = []
        for eps in epses:
            finite_diff = (self.__call__(vol + eps*vol_direction, s=s) - self.__call__(vol, s=s))/eps
            grad = np.sum(self.jacobianT(vol, s=s)*vol_direction)
            r = abs(finite_diff-grad)
            resi.append( r )
            resi_rel.append( r/abs(grad) )

        if plot:
            from plt import plt
            plt.plot(np.array(epses), np.array(resi))
            
        return epses, resi, resi_rel


class diff_KLDivergence(): # convex, separable

    def __init__(self, data, beta):

        #self.dim = dim
        self.data = data
        self.maxdata = np.max(data)
        self.beta = beta
        self.lip = self.maxdata/(self.beta**2)


    def __call__(self, x):

        #with np.errstate(divide='ignore'):
        if np.any(x<0):
            return np.inf
        else:
            return np.sum(x + self.beta - self.data + self.data*np.log(self.data/(x+self.beta)))

    def jacobianT(self, x):

        return 1-self.data/(x+self.beta)

    def prox(self, x, tau):

        d = (x-tau-self.beta)**2 + 4*(x*self.beta + tau*(self.data-self.beta))
        pr = np.zeros_like(x)
        nneg = (d>=0)
        pr[nneg] = (x[nneg] - tau - self.beta + np.sqrt(d[nneg]))/2.
        return pr
    
    def update_lip(self, x):
        return self.maxdata/((self.beta+np.min(x))**2)
    
    def conv_conj_prox(self, x, tau):
        return x - tau*self.prox(x/tau, tau=1./tau)
    
    def prox_convex_dual(self, x, sigma): #should be same
        lam = 1.
        tau = sigma
        d = (x + tau*self.beta - lam)**2 + 4*lam*tau*self.data
        return 0.5*(x + tau*self.beta + lam - np.sqrt(d))
    

class diff_KLDivergence_conj(): # convex, separable

    def __init__(self, data, beta):

        #self.dim = dim
        self.data = data
        self.maxdata = np.max(data)
        self.beta = beta
        self.lip = self.maxdata/(self.beta**2)
    
    def prox(self, x, tau):
        lam = 1.
        d = (x + tau*self.beta - lam)**2 + 4*lam*tau*self.data
        pr = np.zeros_like(x)
        nneg = (d>=0)
        pr[nneg] = 0.5*(x[nneg] + tau*self.beta + lam - np.sqrt(d[nneg]))
        return pr


class conv_conj():
    def __init__(self, operator):

        self.operator = operator
        
    def prox(self, x, tau):
        
        return x - tau*self.operator.prox(x/tau, tau=1./tau)
    
    def prox_convex_dual(self, x, tau):
        
        return self.prox(x, tau)
    
# check prox
#KL = diff_KLDivergence_conj(np.ones(1), backg); KL.data = 1. 
#KL1 = diff_KLDivergence(np.ones(1), backg); KL1.data = 1.
#KL2 = conv_conj(KL1)
#np.max(np.abs(KL.prox(y[0], tau=1.)-KL1.conv_conj_prox(y[0], tau=1.)))
#np.max(np.abs(KL.prox(y[0], tau=1.)-KL2.prox(y[0], tau=1.)))
    

class compose:
    #compose diff (SECOND) with linear (FIRST)
    def __init__(self, FIRST, SECOND):

        #self.lip = FIRST.lip*SECOND.lip
        self.lip = SECOND.lip*(FIRST.lip**2)
        self.FIRST = FIRST
        self.SECOND = SECOND


    def __call__(self, x):
        
        #self.px = self.FIRST(x)
        #return self.SECOND(self.px)
        return self.SECOND(self.FIRST(x))


    def jacobianT(self, x):

        #self.px = self.FIRST(x)
        #return self.FIRST.adjoint( self.SECOND.jacobianT( self.px ) )
        return self.FIRST.adjoint( self.SECOND.jacobianT( self.FIRST(x) ) )
    
    def update_lip(self, x):
        
        #return self.SECOND.update_lip(self.px)*(self.FIRST.lip**2)
        return self.SECOND.update_lip(self.FIRST(x))*(self.FIRST.lip**2)
    
    
class Id_class:
                def __init__(self):
                    pass
                def __call__(self, x):
                    return x
                def adjoint(self, x):
                    return x


class positive_class:
                def __init__(self):
                    pass
                def __call__(self, x):
                    return np.inf if bool(np.any(x<0)) else 1.
                def prox(self, x, tau=1.):
                    return np.maximum(x, 0.)


class weighted_L2_TV_denoiser:

    def __init__(self, prior, nonneg=False, verbose=False):
        
        self.prior = prior
        self.nonneg = nonneg
        self.verbose = verbose

    def init(self, img, weights):
        
        self.img = img
        self.x = self.img.copy()
        self.xbar = self.img.copy()
        self.weights = weights

        self.y = np.zeros((self.x.ndim, ) + self.x.shape,
                                dtype=np.float32)

        if isinstance(weights, np.ndarray):
            self.gam = self.weights.min()
        else:
            self.gam = self.weights

        self.ndim = len([x for x in self.img.shape if x > 1])
        self.grad_op_norm = np.sqrt(self.ndim * 4)

        self.tau = 1.0 / self.gam
        self.sig = 1.0 / (self.tau * self.grad_op_norm**2)

        self.epoch_counter = 0
        self.cost = np.array([], dtype=np.float32)

    def run_update(self):
        # (1) forward model
        self.y += self.sig * self.prior.gradient_operator.__call__(self.xbar)

        # (2) proximity operator
        self.y = self.prior.gradient_norm.prox_convex_dual(self.y,
                                                           sigma=self.sig)

        # (3) adjoint model
        xnew = self.x - self.tau * self.prior.gradient_operator.adjoint(self.y)

        # (4) apply proximity of G
        xnew = (xnew + self.weights * self.img * self.tau) / (
            1.0 + self.weights * self.tau)
        if self.nonneg:
            xnew = np.clip(xnew, 0, None)

        # (5) calculate the new stepsizes
        self.theta = 1.0 / np.sqrt(1 + 2 * self.gam * self.tau)
        self.tau *= self.theta
        self.sig /= self.theta

        # (6) update variables
        self.xbar = xnew + self.theta * (xnew - self.x)
        self.x = xnew.copy()

    def run(self, niter, calculate_cost=False):
        cost = np.zeros(niter, dtype=np.float32)

        for it in range(niter):
            if self.verbose:
                print(f"epoch {self.epoch_counter+1}")
            self.run_update()

            if calculate_cost:
                cost[it] = self.calculate_cost()

            self.epoch_counter += 1

        self.cost = np.concatenate((self.cost, cost))

    def calculate_cost(self):
        return 0.5 * (self.weights *
                      (self.x - self.img)**2).sum() + self.prior(self.x)
                

class GradientNorm:

    def __init__(self, name='l2_l1'):
        self.name = name

        if not self.name in ['l2_l1', 'l2_sq']:
            raise NotImplementedError

    def __call__(self, x):
        if self.name == 'l2_l1':
            n = np.linalg.norm(x, axis=0).sum()
        elif self.name == 'l2_sq':
            n = (x**2).sum()

        return n

    def prox_convex_dual(self, x, sigma=None):
        
        if self.name == 'l2_l1':
            gnorm = np.linalg.norm(x, axis=0)
            r = x / np.clip(gnorm, 1, None)
        elif self.name == 'l2_sq':
            r = x / (1 + sigma)

        return r


class GradientOperator:

    def __init__(self, joint_grad_field=None):

        # e is the normalized joint gradient field that
        # we are only interested in the gradient component
        # perpendicular to it
        self.e = None

        if joint_grad_field is not None:
            norm = np.linalg.norm(joint_grad_field, axis=0)
            inds = np.where(norm > 0)
            self.e = joint_grad_field.copy()

            for i in range(self.e.shape[0]):
                self.e[i,
                       ...][inds] = joint_grad_field[i, ...][inds] / norm[inds]

    def __call__(self, x):
        g = []
        for i in range(x.ndim):
            g.append(np.diff(x, axis=i, append=np.take(x, [-1],
                                                                   i)))
        g = np.array(g)

        if self.e is not None:
            g = g - (g * self.e).sum(0) * self.e

        return g

    def adjoint(self, y):
        d = np.zeros(y[0, ...].shape, dtype=y.dtype)

        if self.e is not None:
            y2 = y - (y * self.e).sum(0) * self.e
        else:
            y2 = y

        for i in range(y.shape[0]):
            d -= np.diff(y2[i, ...],
                               axis=i,
                               prepend=np.take(y2[i, ...], [0], i))

        return d


class GradientBasedPrior:

    def __init__(self, gradient_operator, gradient_norm, beta=1.):
        self.gradient_operator = gradient_operator
        self.gradient_norm = gradient_norm
        self.beta = beta

    def __call__(self, x):
        return float(self.beta *
                     self.gradient_norm(self.gradient_operator.__call__(x)))
        

class GradientOperatorFlat(GradientOperator):
    
    def __init__(self, shp, **kwargs):
        super().__init__(**kwargs)
        self.shp = shp
    
    def __call__(self, x):
        return super().__call__(x.reshape(self.shp))
    
    def adjoint(self, y):
        return super().adjoint(y).ravel()
    

class l12_norm():
    def __init__(self, shp):
        self.shp = shp
        self.size = int(np.prod(np.asarray(shp))) #np.prod(shp)
        self.ndim = len(shp)
    
    def __call__(self, x):
        return np.sum(self.l2(x))
        
    def project(self, x):    
        n = self.l2(x).clip(1.) #np.maximum(self.l2(x),1.)
        return (x.reshape(self.ndim,-1)/n).ravel() #x/(np.broadcast_to(n, (self.ndim, n.size)).flat) #(x.reshape(self.ndim,-1)/n).ravel()
    
    def l2(self,x):
        return np.linalg.norm(x.reshape(self.ndim,-1), axis=0)


class TV_cost():

    def __init__(self, shp, n_prox_iter, lam=1., proj=None, iter_func=None, acc='BT', dtype=np.float64):
        self.shp = shp
        self.size = int(np.prod(np.asarray(shp))) #np.prod(shp)
        self.ndim = len(shp)
        self.lam = lam #reg constant
        self.n_prox_iter = n_prox_iter

        self.Grad = LinOpGrad(shp, dtype=dtype) #should reshape like .reshape(ndim,-1), but when? proj should take this into account.
        self.norm = l12_norm(shp)
        self.iter_func = lambda x: x if iter_func is None else iter_func
        self.acc = acc

        self.tv_ct = 8. #12. for 3D

        if proj is None:
            self.proj = lambda x: x
        elif isinstance(proj, (list, np.ndarray)): #interpret as a minmax constrain
            self.proj = lambda x: x.clip(*proj)
        elif callable(proj): #hasattr(proj, '__call__'):
            self.proj = proj
        else:
            raise Exception('wrong type')
        #self.proj_norm to add to call eventually


    def __call__(self, x):
        return self.lam*self.norm(self.Grad(x))
                               #l2 isotropic, l1 for anisotropic (no need to reshape for l1 because independent)
    def prox(self, y, tau):
        tau = self.lam*tau

        ilip = 1./(self.tv_ct*tau)
        x = np.zeros(self.size*self.ndim, dtype=y.dtype)
        x_prev = np.zeros_like(x)
        t_prev = 1.

        for i in range(self.n_prox_iter):
            x_temp = x + ilip*self.Grad(\
                               self.proj( \
                                   y-tau*self.Grad.adjoint(x)
                                ) )
            
            x_temp = self.norm.project(x_temp)

            if self.acc=="BT":
                t = 0.5 * (1 + math.sqrt(1 + 4 * t_prev ** 2))
            else:
                t = 1
            x = x_temp + (x_temp - x_prev) * (t_prev - 1) / t

            x_prev = x_temp
            t_prev = t
            self.n_prox_iter = self.iter_func(self.n_prox_iter)

        return self.proj(y-tau*self.Grad.adjoint(x_temp)) 
    

class LinOpGrad():
    
    def __init__(self, shp, dtype=np.float64):
        import pylops as pl #pylops supports both numpy and cupy, pylops_gpu for pytorch

        self.shp = shp
        self.ndim = len(shp)
        self.Grad = pl.Gradient(shp, kind="forward", dtype=dtype) # required flattened inputs for evaluations and returns flattened outputs
        # This class does nothing but change rmatvec for adjoint

    def __call__(self, x):
        return self.Grad.matvec(x) #self.Grad(x)
        
    def adjoint(self, x):        
        return self.Grad.rmatvec(x) #self.Grad.adjoint(x) to create hermitian adjoint