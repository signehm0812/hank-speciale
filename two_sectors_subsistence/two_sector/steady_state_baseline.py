# find steady state

import time
import numpy as np
from scipy import optimize

from consav import elapsed

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst

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
        n = 1.0*z

        c = (1+ss.r)*par.a_grid + ss.w*n + T
        va[0,i_z,:] = c**(-par.sigma)

    ss.vbeg_a[0] = ss.z_trans[0]@va[0]
        
def evaluate_ss(model,do_print=False):
    """ evaluate steady state"""

    par = model.par
    ss = model.ss

    # a. fixed
    ss.pi = 0.0
    ss.P = 1.0
    ss.Y = 1.0
    ss.Y_star = 1.0
    ss.pm = 1.0
    
    # b. targets
    ss.r = par.r_target_ss
    ss.A = ss.B = par.B_target_ss
    ss.G = par.G_target_ss

    # c. monetary policy
    ss.i = ss.istar = (1+ss.r)*(1+ss.pi)-1

    # d. firms
    ss.w = ((par.mu**(par.gamma-1)-par.alpha*ss.pm**(1-par.gamma))/(1-par.alpha))**(1/(1-par.gamma))*ss.Z
    ss.mc = ((1-par.alpha)*(ss.w/ss.Z)**(1-par.gamma)+par.alpha*ss.pm**(1-par.gamma))**(1/(1-par.gamma))
    ss.N = (1-par.alpha)*(ss.w/ss.mc)**(-par.gamma)*ss.Z**(par.gamma-1)*ss.Y
    ss.M = par.alpha*(ss.pm/ss.mc)**(-par.gamma)*ss.Y
    ss.adjcost = 0.0
    ss.d = ss.Y-ss.w*ss.N-ss.pm*ss.M-ss.adjcost
    
    # e. government
    ss.tau = ss.r*ss.B + ss.G + par.chi

    #print(f'Z = {ss.Z:.4f},\t M = {ss.M:.4f},\t beta = {par.beta:.4f},\t N = {ss.N:.4f}') #Print so we can see what goes wrong if root solving doesnt converge


    # f. household 
    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)

    # g. market clearing
    ss.C = ss.Y-ss.G-ss.adjcost-ss.pm*ss.M

def objective_ss(x,model,do_print=False):
    """ objective function for finding steady state """

    par = model.par
    ss = model.ss

    ss.Z = x[0]
    par.beta = x[1]
    #par.varphi = x[1]
    
    if par.beta <=0.94: par.beta = 0.94

    if par.beta > 1/(1+ss.r): par.beta = 1/(1+ss.r)

    evaluate_ss(model,do_print=do_print)
    
    return np.array([ss.A_hh-ss.B,ss.N_hh-ss.N])

def find_ss(model,do_print=False):
    """ find the steady state """

    par = model.par
    ss = model.ss

    # a. find steady state
    t0 = time.time()

    # set initial values:
    ss.Z = 0.5

    res = optimize.root(objective_ss,[ss.Z, par.beta],method='hybr',tol=par.tol_ss,args=(model))

    # final evaluation
    objective_ss(res.x,model)

    # b. print
    if do_print:

        print(f'steady state found in {elapsed(t0)}')
        print(f' Z   = {res.x[0]:8.4f}')
        print(f' beta   = {res.x[1]:8.4f}')       
        print(f' M   = {ss.M:8.4f}')          
        print(f' N   = {ss.N:8.4f}')          
        print(f' HH_ell   = {ss.ELL_hh:8.4f}')  
        print(f' wage  = {ss.w:8.4f}')    
        print(f' par.varphi   = {par.varphi:8.4f}')
        print(f'Discrepancy in B = {ss.A-ss.A_hh:12.8f}')
        print(f'Discrepancy in C = {ss.C-ss.C_hh:12.8f}')
        print(f'Discrepancy in N = {ss.N-ss.N_hh:12.8f}')
