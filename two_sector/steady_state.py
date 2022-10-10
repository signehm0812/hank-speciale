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
    
    ss.ell_N = par.ell_N_target
    ss.ell_L = par.ell_L_target

    for i_z in range(par.Nz):

        z = par.z_grid[i_z]
        T = (ss.d_N+ss.d_L)*z - ss.tau*z
        n_N = ss.ell_N*z 
        n_L = ss.ell_L*z

        c = (1+ss.r)*par.a_grid + (ss.w_N*n_N+ss.w_L*n_L) + T
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
    ss.ell_N = par.ell_N_target
    ss.ell_L = par.ell_L_target
    
    # b. targets
    ss.r = par.r_target_ss
    ss.A = ss.B = par.B_target_ss #should be ss.A_hh?
    ss.G = par.G_target_ss

    # c. monetary policy
    ss.i = ss.r = ss.istar = 0.0

    # d. firms
    ss.Y_N = (par.alpha_N**(1/par.gamma_N)*ss.M_N**((par.gamma_N-1)/par.gamma_N)+(1-par.alpha_N)**(1/par.gamma_N)*(ss.Z*ss.N_N)**((par.gamma_N-1)/par.gamma_N))**(par.gamma_N/(par.gamma_N-1))
    ss.w_N = ((par.mu_N**(par.gamma_N-1)-par.alpha_N*par.pm**(1-par.gamma_N))*(ss.Z**par.gamma_N/(1-par.alpha_N)))**(1/(1-par.gamma_N))
    ss.d_N = ss.Y_N-ss.w_N*ss.N_N-par.pm*ss.M_N-ss.adjcost_N
    ss.mc_N = ((1-par.alpha_N)*(ss.w_N*ss.Z)**(1-par.gamma_N)+par.alpha_N*par.pm**(1-par.gamma_N))**(1/(1-par.gamma_N))
    #ss.M_N = (par.pm/ss.mc_N)**(-par.gamma_N)*par.alpha_N*ss.Y_N
    ss.adjcost_N = 0.0
    
    ss.Y_L = (par.alpha_L**(1/par.gamma_L)*ss.M_L**((par.gamma_L-1)/par.gamma_L)+(1-par.alpha_L)**(1/par.gamma_L)*(ss.Z*ss.N_L)**((par.gamma_L-1)/par.gamma_L))**(par.gamma_L/(par.gamma_L-1))
    ss.w_L = ((par.mu_L**(par.gamma_L-1)-par.alpha_L*par.pm**(1-par.gamma_L))*(ss.Z**par.gamma_L/(1-par.alpha_L)))**(1/(1-par.gamma_L))
    ss.d_L = ss.Y_L-ss.w_L*ss.N_L-par.pm*ss.M_L-ss.adjcost_L
    ss.mc_L = ((1-par.alpha_L)*(ss.w_L*ss.Z)**(1-par.gamma_L)+par.alpha_L*par.pm**(1-par.gamma_L))**(1/(1-par.gamma_L))
    #ss.M_L = (par.pm/ss.mc_L)**(-par.gamma_L)*par.alpha_L*ss.Y_L
    ss.adjcost_L = 0.0

    ss.adjcost = ss.adjcost_N + ss.adjcost_L
    ss.Y = ss.Y_N + ss.Y_L
    ss.M = ss.M_N + ss.M_L
    ss.N = ss.N_N + ss.N_L

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

    ss.M_N = x[0]
    ss.M_L = x[1]

    evaluate_ss(model,do_print=do_print)
    
    return np.array([ss.A_hh-ss.B]) #,ss.N_hh-ss.N

def find_ss_direct(model,do_print=False,M_min=1.0,M_max=10.0,NK=10):
    
    t0 = time.time()

    if do_print: print(f'### step 1: broad search ###\n')
    M_ss_vec = np.linspace(M_min,M_max,NK)
    clearing_A = np.zeros(M_ss_vec.size) # asset market errors

    for i,M_ss in enumerate(M_ss_vec):
        
        try:
            clearing_A[i] = objective_ss(M_ss,model,do_print=do_print)
        except Exception as e:
            clearing_A[i] = np.nan
            print(f'{e}')
            
        if do_print: print(f'clearing_A = {clearing_A[i]:12.8f}\n')
            
    # b. determine search bracket
    if do_print: print(f'### step 2: determine search bracket ###\n')

    M_max = np.min(M_ss_vec[clearing_A < 0])
    M_min = np.max(M_ss_vec[clearing_A > 0])

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
    res = optimize.root(objective_ss,[ss.M_N,ss.M_L],method='hybr',tol=par.tol_ss,args=(model))

    # final evaluation
    objective_ss(res.x,model)

    # b. print
    if do_print:

        print(f'steady state found in {elapsed(t0)}')
        print(f' M_N   = {res.x[0]:8.4f}')
        print(f' M_L = {res.x[1]:8.4f}')
        print('')
        print(f'Discrepancy in B = {ss.A-ss.A_hh:12.8f}')
        print(f'Discrepancy in C = {ss.C-ss.C_hh:12.8f}')
        print(f'Discrepancy in N = {ss.N-ss.N_hh:12.8f}')
