from argeoPET import array_lib as np


##------------------------------------------------------##
### OSEM 
# solves KL alone
def OSEM(Ps, init=None, maxiter=70, comp=None): #comp=1e5
    vol = np.ones(Ps.vol_dim, dtype=np.float32) if init is None else np.copy(init)
    maxiter = 70
    
    if comp is None:
        comp = 1e5 #np.finfo(vol.dtype).eps*1e4*Ps.vol_dim
        Ps.create_norm_adjoints()
        sensi_image = Ps.Ps_na[ind]+comp
    else:
        sensi_image = np.maximum(Ps.pt1, comp)
    

    for k in range(maxiter):
        ind = int(np.random.randint(0, Ps.nsubsets))
        vol *= Ps.adjoint(Ps.data(ind) / (Ps(vol, ind=ind) + Ps.backg(ind)), ind=ind) / sensi_image
        
    return vol


##------------------------------------------------------##
### OSEM-TV
# key is in the modified TV prox with weights
def OSEM_TV(Ps, init=None, maxiter = 70, maxiter_prox = 30, lam = 1750., comp=None, force_full=False):
    vol = np.ones(Ps.vol_dim, dtype=np.float32) if init is None else np.copy(init)
 
    lb = np.finfo(vol.dtype).eps; ub = np.finfo(vol.dtype).max
    if comp is None:
        comp = np.finfo(vol.dtype).eps*1e4*Ps.vol_dim
        sensi_image = Ps.pt1 + comp #np.maximum(Ps.pt1, comp)
    else:
        sensi_image = np.maximum(Ps.pt1, comp)
    
    from functional_classes import GradientBasedPrior, GradientOperator, GradientNorm, weighted_L2_TV_denoiser
    prior = GradientBasedPrior(GradientOperator(), GradientNorm(name="l2_l1"), lam)
    denoiser = weighted_L2_TV_denoiser(prior, verbose=False, nonneg=True)

    for k in range(maxiter):
        
        # OSEM
        if not force_full:
            ind = int(np.random.randint(0, Ps.nsubsets))
            vol *= Ps.adjoint(Ps.data(ind) / (Ps(vol, ind=ind) + Ps.backg(ind)), ind=ind) / sensi_image #/ (Ps.Ps_na[ind]+comp) #/ sensi_image
        else:
            vol *= Ps.adjoint(Ps.data(np.s_[:]) / (Ps(vol, s=np.s_[:]) + Ps.backg(np.s_[:])), s=np.s_[:]) / sensi_image

        # WEIGHTS
        weights = np.clip(sensi_image/np.maximum(lam*vol, lb), lb, ub)
        
        # SOLVE PROX
        denoiser.init(vol.reshape(Ps.vol_shp), weights.reshape(Ps.vol_shp))
        denoiser.run(maxiter_prox, calculate_cost=False)
        vol = denoiser.x.ravel().copy()
        
    return vol


##------------------------------------------------------##
### CLASS FOR GENERAL ACCELERATED PROXIMAL GRADIENT DESCENT
import math
class updater():
    def __init__(self, acc_method="BT", p=1., q=1., r=4., d=50.,
                       momentum_method=None, s=1.,
                       restart_method=None, xi=1., restart_t=True): 
                       # p,q>0; 0<r<=4; 2,10<d<80; 0<xi<=1.; s is float or class

        if momentum_method=="SC" and all(hasattr(s,  attr) for attr in ["lip", "eigen_min"]):
            sratio = math.sqrt(s.eigen_min/s.lip)
            s = (1-sratio)/(1+sratio)

        self.r = r
        self.s = s
        self.restart_method = restart_method
        
        self.acc_dict = {
                    "BT": lambda i, t_prev:
                          0.5 * (p + math.sqrt(q + self.r * t_prev ** 2)),
                    "CD": lambda i, t_prev:
                          (i + d) / d,
                                     }

        self.mom_dict = {
                    "SC": lambda t, t_prev:
                          self.s,
                                     }

        self.rest_dict = {
                    "RESTART": lambda t, x_star, x_prev, y:
                                (t, x_star) if np.sum((y-x_star)*(x_star-x_prev))<0 
                                else (1 if restart_t else t, [x_prev, self.upr(xi*self.r)][0]),
                                     }

        self.acc = self.acc_dict.get(acc_method,     lambda i, t_prev: t_prev)
        self.mom = self.mom_dict.get(momentum_method,       lambda t, t_prev: (t_prev - 1.) / t)
        self.rest = self.rest_dict.get(restart_method,  lambda t, x_star, x_prev, y: (t, x_star))

    def accelerate(self, i, t_prev):
        return self.acc(i, t_prev)

    def momentum(self, t, t_prev):
        return self.mom(t, t_prev)

    def maybe_restart(self, t, x_star, x_prev, y):
        return self.rest(t, x_star, x_prev, y)

    def upr(self, new_r):
        self.r = new_r


def print_cost(F, G, x, verbose=True):
    sum_costs = F(x) + G(x)
    if verbose: print(sum_costs)
    return sum_costs.item()


##------------------------------------------------------##
### GENERAL ACCELERATED PROXIMAL GRADIENT DESCENT
def APGD(F, G, x0=None, niter=20, verbose=0, alpha=1., u=None, update_ilip=None, converged=None, **kwargs):

    #SORT OUT ARGS
    if converged is None: converged = lambda a, b: False
    if u is None: u = updater(**kwargs)
    restart_enabled = False if u.restart_method is None else True
    

    #LIPS
    ilip = alpha/F.lip
    if update_ilip is True: update_ilip = lambda k, il, x: alpha/F.update_lip(x)
    if update_ilip is None: update_ilip = lambda k, il, x: ilip
    

    #INIT
    y = np.zeros(F.dim[0]) if x0 is None else np.copy(x0)
    x_prev = np.zeros_like(y)
    t_prev = 1
    costs = []


    #RUN
    itr = 1
    while True:

        #DESCEND, PRINT AND CHECK CONVERGENCE
        x_star = G.prox(y - ilip * F.jacobianT(y), tau=ilip)

        if verbose and (itr%verbose==0): costs.append( print_cost(F, G, x_star) )
        if itr>=niter or converged(x_star, x_prev): break #for/while not finished

        #ACCELERATE
        t = u.accelerate(itr, t_prev)
        if restart_enabled: t, x_star = u.maybe_restart(t, x_star, x_prev, y)
        y = x_star + u.momentum(t, t_prev)*(x_star - x_prev)

        #RENAME ITERANDS
        x_prev, t_prev = x_star, t
        ilip = update_ilip(itr, ilip, x_star)
        itr+=1


    #DONE    
    if verbose: 
        from plt import plt
        plt.plot(verbose*np.arange(len(costs)), costs); plt.show()
    return x_star #, costs, itr<niter


##------------------------------------------------------##
### GENERAL GRADIENT DESCENT
def GD(C, min_max=None, **kwargs): 

    box_project = (lambda y : y) if min_max is None else (lambda y : y.clip(*min_max))

    class indicator():
        def __call__(self, x):
            return 0 #clipping anyway
        
        def prox(self, y, tau):
            return box_project(y)
   
    return APGD(C, indicator(), **kwargs)