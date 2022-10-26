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
        T = (ss.d_N+ss.Q*ss.d_L)*z - ss.tau*z
        n = 1.0*z

        c = (1+ss.r)*par.a_grid + ss.w_N*n + T
        va[0,i_z,:] = c**(-par.sigma)

    ss.vbeg_a[0] = ss.z_trans[0]@va[0]
        
def evaluate_ss(model,do_print=False):
    """ evaluate steady state"""

    par = model.par
    ss = model.ss

    # a. fixed
    #ss.Z = 1.0
    #ss.N = 1.0
    ss.pm = 1.0
    ss.pi_N = 0.0
    ss.pi_L = 0.0
    ss.Y = 1.0
    ss.Z_L = 1.0

    # b. targets
    ss.r = par.r_target_ss
    ss.A = ss.B = par.B_target_ss
    ss.G = par.G_target_ss

    # c.. monetary policy
    ss.i = ss.r = ss.istar = 0.0

    # d. firms  
    P = (par.alpha_hh+ss.Q**(1-par.gamma_hh)*(1-par.alpha_hh))**(1/(1-par.gamma_hh)) 
    ss.w_L = (ss.Z_L)*((par.mu_L**(par.gamma_L-1)-par.alpha_L*ss.pm**(1-par.gamma_L))/(1-par.alpha_L))**(1/(1-par.gamma_L))
    ss.w_N = ss.Q*ss.w_L
    ss.Z_N = ss.w_N/((par.mu_N**(par.gamma_N-1)-par.alpha_L*ss.pm**(1-par.gamma_N))/(1-par.alpha_N))**(1/(1-par.gamma_N))
    ss.Y_N = ss.Y*P-ss.Q*ss.Y_L
    ss.mc_N = ((1-par.alpha_N)*(ss.w_N/ss.Z_N)**(1-par.gamma_N)+par.alpha_N*ss.pm**(1-par.gamma_N))**(1/(1-par.gamma_N))
    ss.mc_L = ((1-par.alpha_L)*(ss.w_L/ss.Z_L)**(1-par.gamma_L)+par.alpha_L*ss.pm**(1-par.gamma_L))**(1/(1-par.gamma_L))
    ss.M_L = (par.alpha_L*(ss.pm/ss.mc_L)**(-par.gamma_L)*ss.Y_L)#/ss.Q    
    ss.N_L = (1-par.alpha_L)*(ss.w_L/ss.mc_L)**(-par.gamma_L)*ss.Z_L**(par.gamma_L-1)*ss.Y_L    
    ss.M_N = par.alpha_N*(ss.pm/ss.mc_N)**(-par.gamma_N)*ss.Y_N    
    ss.N_N = (1-par.alpha_N)*(ss.w_N/ss.mc_N)**(-par.gamma_N)*ss.Z_N**(par.gamma_N-1)*ss.Y_N
    ss.adjcost_L = 0.0
    ss.adjcost_N = 0.0
    ss.d_N = ss.Y_N-ss.w_N*ss.N_N-ss.pm*ss.M_N-ss.adjcost_N
    ss.d_L = ss.Y_L-ss.w_L*ss.N_L-ss.pm*ss.M_L-ss.adjcost_L
    ss.N = ss.N_N+ss.N_L

    print(f'Z_N = {ss.Z_N:.4f},\t Z_L = {ss.Z_L:.4f},\t Q = {ss.Q:.4f},\t M_N = {ss.M_N:.4f},\t M_L = {ss.M_L:.4f},\t beta = {par.beta:.4f},\t N_N = {ss.N_N:.4f},\t N_L = {ss.N_L:.4f}') #Print so we can see what goes wrong if root solving doesnt converge
    
    # e. government
    ss.tau = ss.r*ss.B + ss.G

    # f. household 
    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)

    # g. market clearing
    ss.C_N = ss.Y_N-ss.adjcost_N-ss.pm*ss.M_N
    ss.C_L = ss.Y_L-ss.adjcost_L-ss.pm*ss.M_L
    
    ss.C = (ss.C_N + ss.Q*ss.C_L)/P
    
    #ss.C = ss.Y_N-ss.adjcost_N-ss.pm*ss.M_N + ss.Q*(ss.Y_L-ss.adjcost_L-ss.pm*ss.M_L)

def objective_ss(x,model,do_print=False):
    """ objective function for finding steady state """

    par = model.par
    ss = model.ss

    #ss.M_L = x[0]
    #ss.N_L = x[1]
    ss.Y_L = x[0]
    par.beta = x[1]
    ss.Q = x[2]
    #par.varphi = x[3]

    if ss.Q <= 0: ss.Q = 0.1

    if ss.Q > 5: ss.Q = 5.0

    if ss.Y_L <= 0.1: ss.Y_L = 0.1

    if ss.Y_L > 5: ss.Y_L = 5.0

    if par.beta <=0.94: par.beta = 0.94

    if par.beta > 1.0: par.beta = 1.0

    #if par.varphi <=0.5: par.varphi = 0.5

    #if par.varphi > 10.0: par.varphi = 10.0    

    evaluate_ss(model,do_print=do_print)

    #
    #par.varphi = ss.w_N/(ss.ELL_hh**(par.nu)*((ss.C**par.sigma)*ss.C_HAT_N_hh/par.alpha_hh)**(1/par.gamma_hh))
    #*((par.mu_L**(par.gamma_L-1)-par.alpha_L*ss.pm**(1-par.gamma_L))/(1-par.alpha_L))**(1/(1-par.gamma_L))*ss.Q*ss.Z_L
        
    #if par.varphi >= 1.9: par.varphi = 1.9


    #return np.array([ss.A_hh-ss.B])
    return np.array([ss.A_hh-ss.B,ss.N_hh-ss.N,ss.C_N_hh-ss.C_N])

def find_ss(model,do_print=False):
    """ find the steady state """

    par = model.par
    ss = model.ss

    # a. find steady state
    t0 = time.time()

    #Set initial values for ss.Z_L and ss.Q before looping over
    ss.Y_L = 0.5
#   ss.N_L = 0.5
    ss.Q = 0.5

    #res = optimize.root(objective_ss,[par.beta, par.varphi],method='hybr',tol=par.tol_ss,args=(model))
    res = optimize.root(objective_ss,[ss.Y_L,par.beta,ss.Q],method='hybr',tol=par.tol_ss,args=(model))

#    par.varphi = ss.ELL_hh**(-par.nu)*((ss.C**par.sigma)*ss.C_HAT_N_hh/par.alpha_hh)**(-1/par.gamma_hh)*((par.mu_L**(par.gamma_L-1)-par.alpha_L*ss.pm**(1-par.gamma_L))/(1-par.alpha_L))**(1/(1-par.gamma_L))*ss.Q/ss.Z_L
        
#    if par.varphi >= 2.0: par.varphi = 2.0

    # final evaluation
    objective_ss(res.x,model)

    # b. print
    if do_print:

        print(f'steady state found in {elapsed(t0)}')
        #print(f' M_N   = {res.x[0]:8.4f}')
        #print(f' beta   = {res.x[1]:8.4f}')
        print(f' Z_N   = {ss.Z_N:8.4f}')
        print(f' Z_L   = {ss.Z_L:8.4f}')        
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
        print(f'Discrepancy in C_L = {ss.C_L-ss.C_L_hh:12.8f}')
        print(f'Discrepancy in C_N = {ss.C_N-ss.C_N_hh:12.8f}')
        print(f'Discrepancy in N = {ss.N_L+ss.N_N-ss.N_hh:12.8f}')
