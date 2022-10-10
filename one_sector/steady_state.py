# find steady state

import time
import numpy as np
from scipy import optimize

from consav import elapsed

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst

import root_finding

def prepare_hh_ss(model):
    """ prepare the household block for finding the steady state """

    par = model.par
    ss = model.ss

    ##################################
    # 1. grids and transition matrix #
    ##################################

    # b. a
    par.a_grid[:] = equilogspace(par.a_min,par.a_max,par.Na)

    # c. z
    par.z_grid[:],ss.z_trans[0,:,:],e_ergodic,_,_ = log_rouwenhorst(par.rho_z,par.sigma_psi,n=par.Nz)

    ###########################
    # 2. initial distribution #
    ###########################
    
    for i_fix in range(par.Nfix):
        ss.Dz[i_fix,:] = e_ergodic/par.Nfix
        ss.Dbeg[i_fix,:,0] = ss.Dz[i_fix,:]
        ss.Dbeg[i_fix,:,1:] = 0.0    

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    va = np.zeros((par.Nfix,par.Nz,par.Na))

    for i_z in range(par.Nz):

        z = par.z_grid[i_z]
        T = ss.d*z - ss.tau*z
        n = ss.ell*z 

        c = (1+ss.r)*par.a_grid + (ss.w*n) + T
        va[0,i_z,:] = c**(-par.sigma)

    ss.vbeg_a[0] = ss.z_trans[0]@va[0]
        
def evaluate_ss(model,do_print=False):
    """ evaluate steady state"""

    par = model.par
    ss = model.ss

    # a. fixed
    ss.Z = 1.0
    ss.N = 1.0
    ss.pi = 0.0
    ss.ell = par.ell_target
    
    # b. targets
    ss.r = par.r_target_ss
    ss.A = ss.B = par.B_target_ss #should be ss.A_hh?
    ss.G = par.G_target_ss

    # c. monetary policy
    ss.i = ss.r = ss.istar = 0.0

    # d. firms
    ss.Y = (par.alpha**(1/par.gamma)*ss.M**((par.gamma-1)/par.gamma)+(1-par.alpha)**(1/par.gamma)*(ss.Z*ss.N)**((par.gamma-1)/par.gamma))**(par.gamma/(par.gamma-1))
    ss.w = ((par.mu**(par.gamma-1)-par.alpha*par.pm**(1-par.gamma))*(ss.Z**par.gamma/(1-par.alpha)))**(1/(1-par.gamma))
    ss.d = ss.Y-ss.w*ss.N-par.pm*ss.M-ss.adjcost
    ss.mc = ((1-par.alpha)*(ss.w*ss.Z)**(1-par.gamma)+par.alpha*par.pm**(1-par.gamma))**(1/(1-par.gamma))
    ss.adjcost = 0.0
    
    # e. government
    ss.tau = ss.r*ss.B + ss.G

    # f. household 
    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)

    # g. market clearing
    ss.C = ss.Y-ss.G-ss.adjcost-ss.M*par.pm

def objective_ss(x,model,do_print=False):
    """ objective function for finding steady state """

    par = model.par
    ss = model.ss

    ss.M = x[0]

    evaluate_ss(model,do_print=do_print)
    
    return np.array([ss.A_hh-ss.B]) #,ss.N_hh-ss.N


def find_ss_direct(model,do_print=False,M_min=1.0,M_max=10.0,NK=10):
    
    t0 = time.time()

    if do_print: print(f'### step 1: broad search ###\n')
    M_ss_vec = np.linspace(M_min,M_max,NK)
    clearing_AB = np.zeros(M_ss_vec.size) # asset market errors

    for i,M_ss in enumerate(M_ss_vec):
        
        try:
            clearing_AB[i] = objective_ss(M_ss,model,do_print=do_print)
        except Exception as e:
            clearing_AB[i] = np.nan
            print(f'{e}')
            
        if do_print: print(f'clearing_AB = {clearing_AB[i]:12.8f}\n')
            
    # b. determine search bracket
    if do_print: print(f'### step 2: determine search bracket ###\n')

    M_max = np.min(M_ss_vec[clearing_AB < 0])
    M_min = np.max(M_ss_vec[clearing_AB > 0])

    if do_print: print(f'M in [{M_min:12.8f},{M_max:12.8f}]\n')

    # c. search
    if do_print: print(f'### step 3: search ###\n')

    root_finding.brentq(
        objective_ss,M_min,M_max,args=(model,),do_print=do_print,
        varname='M_ss',funcname='A_hh-B'
    )

    if do_print: print(f'found steady state in {elapsed(t0)}')

def find_ss(model,do_print=False):
    """ find the steady state """

    par = model.par
    ss = model.ss

    # a. find steady state
    t0 = time.time()
    res = optimize.root(objective_ss,ss.M,method='hybr',tol=par.tol_ss,args=(model))

    # final evaluation
    objective_ss(res.x,model)

    # b. print
    if do_print:

        print(f'steady state found in {elapsed(t0)}')
        print(f' M   = {res.x[0]:8.4f}')
        #print(f' M_L = {res.x[1]:8.4f}')
        print('')
        print(f'Discrepancy in B = {ss.A-ss.A_hh:12.8f}')
        print(f'Discrepancy in C = {ss.C-ss.C_hh:12.8f}')
        print(f'Discrepancy in N = {ss.N-ss.N_hh:12.8f}')
