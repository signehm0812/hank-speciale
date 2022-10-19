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
        T = (ss.d_N+ss.d_L)*z - ss.tau*z
        n_N = 0.5*z
        n_L = 0.5*z

        c = (1+ss.r)*par.a_grid + (ss.w_N*n_N+ss.w_L*n_L) + T
        va[0,i_z,:] = c**(-par.sigma)

    ss.vbeg_a[0] = ss.z_trans[0]@va[0]
        
def evaluate_ss(model,do_print=False):
    """ evaluate steady state"""

    par = model.par
    ss = model.ss

    # a. fixed
    ss.Z = 1.0
    ss.N_N = 0.5
    ss.N_L = 0.5
    ss.pm = 1.5    
    ss.pi_N = 0.0
    ss.pi_L = 0.0
    
    # b. targets
    ss.r = par.r_target_ss
    ss.A = ss.B = par.B_target_ss
    ss.G = par.G_target_ss

    # c.. monetary policy
    ss.i = ss.r = ss.istar = 0.0


    # d. firms    
    ss.Y_L = (par.alpha_L**(1/par.gamma_L)*ss.M_L**((par.gamma_L-1)/par.gamma_L)+(1-par.alpha_L)**(1/par.gamma_L)*(ss.Z*ss.N_L)**((par.gamma_L-1)/par.gamma_L))**(par.gamma_L/(par.gamma_L-1))
    ss.w_L = ((par.mu_L**(par.gamma_L-1)-par.alpha_L*ss.pm**(1-par.gamma_L))*(ss.Z**par.gamma_L/(1-par.alpha_L)))**(1/(1-par.gamma_L))
    ss.d_L = ss.Y_L-ss.w_L*ss.N_L-ss.pm*ss.M_L-ss.adjcost_L
    ss.mc_L = ((1-par.alpha_L)*(ss.w_L*ss.Z)**(1-par.gamma_L)+par.alpha_L*ss.pm**(1-par.gamma_L))**(1/(1-par.gamma_L))
    ss.adjcost_L = 0.0
    
    ss.Y_N = (par.alpha_N**(1/par.gamma_N)*ss.M_N**((par.gamma_N-1)/par.gamma_N)+(1-par.alpha_N)**(1/par.gamma_N)*(ss.Z*ss.N_N)**((par.gamma_N-1)/par.gamma_N))**(par.gamma_N/(par.gamma_N-1))
    ss.w_N = ((par.mu_N**(par.gamma_N-1)-par.alpha_N*ss.pm**(1-par.gamma_N))*(ss.Z**par.gamma_N/(1-par.alpha_N)))**(1/(1-par.gamma_N))
    ss.d_N = ss.Y_N-ss.w_N*ss.N_N-ss.pm*ss.M_N-ss.adjcost_N
    ss.mc_N = ((1-par.alpha_N)*(ss.w_N*ss.Z)**(1-par.gamma_N)+par.alpha_N*ss.pm**(1-par.gamma_N))**(1/(1-par.gamma_N))
    ss.adjcost_N = 0.0

    print(ss.M_N, ss.M_L, par.beta, par.varphi) #Print so we can see what goes wrong if root solving doesnt converge

    ss.adjcost = ss.adjcost_N + ss.adjcost_L
    ss.Y = ss.Y_N + ss.Y_L
    ss.N = ss.N_N + ss.N_L
    ss.M = ss.M_N + ss.M_L
    
    # e. government
    ss.tau = ss.r*ss.B + ss.G
    # f. household 
    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)

    # g. market clearing
    ss.C = ss.Y-ss.G-ss.adjcost-ss.pm*ss.M

def objective_ss(x,model,do_print=False):
    """ objective function for finding steady state """

    par = model.par
    ss = model.ss

    ss.M_N = x[0]
    ss.M_L = x[1]
    par.beta = x[2]

    evaluate_ss(model,do_print=do_print)
    
    #return np.array([ss.A_hh-ss.B])
    return np.array([ss.A_hh-ss.B,ss.M_L-((par.alpha_L/par.alpha_N)*((ss.mc_L**par.gamma_N)/(ss.mc_N**par.gamma_N))*(ss.Y_L/ss.Y_N))*ss.M_N,ss.N_hh-ss.N])

def find_ss(model,do_print=False):
    """ find the steady state """

    par = model.par
    ss = model.ss

    # a. find steady state
    t0 = time.time()

    #Set initial values for ss.M_N and ss.M_L before looping over
    ss.M_N = 1.0 
    ss.M_L = 1.0

    #res = optimize.root(objective_ss,[par.beta, par.varphi],method='hybr',tol=par.tol_ss,args=(model))
    res = optimize.root(objective_ss,[ss.M_N,ss.M_L,par.beta],method='hybr',tol=par.tol_ss,args=(model))

    # final evaluation
    objective_ss(res.x,model)

    # b. print
    if do_print:

        print(f'steady state found in {elapsed(t0)}')
        #print(f' M_N   = {res.x[0]:8.4f}')
        #print(f' beta   = {res.x[1]:8.4f}')
        print(f' M_N   = {ss.M_N:8.4f}')
        print(f' M_L   = {ss.M_L:8.4f}')          
        print(f' HH_ell   = {ss.ELL_hh:8.4f}')  
        print(f' wage N  = {ss.w_N:8.4f}')    
        print(f' wage L  = {ss.w_L:8.4f}')                  
        print(f' par.varphi   = {par.varphi:8.4f}')
        print(f' par.beta   = {par.beta:8.4f}')                
        print('')
        print(f'Discrepancy in B = {ss.A-ss.A_hh:12.8f}')
        print(f'Discrepancy in C = {ss.C-ss.C_hh:12.8f}')
        print(f'Discrepancy in N = {ss.N-ss.N_hh:12.8f}')