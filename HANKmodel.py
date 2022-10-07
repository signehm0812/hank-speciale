import numpy as np

from EconModel import EconModelClass
from GEModelTools import GEModelClass

import steady_state
import household_problem
import blocks

class HANKModelClass(EconModelClass,GEModelClass):    

    def settings(self):
        """ fundamental settings """

        # a. namespaces (typically not changed)
        self.namespaces = ['par','ini','ss','path','sim'] # not used today: 'ini', 'path', 'sim'

        # not used today: .sim and .path
        
        # b. household
        self.grids_hh = ['a'] # grids
        self.pols_hh = ['a'] # policy functions
        self.inputs_hh = ['r','tau','w_L','w_N','d_L','d_N'] #['r','tau','w_L','w_N','d_L','d_N'] # direct inputs
        self.inputs_hh_z = [] # transition matrix inputs (not used)
        self.outputs_hh = ['a','c'] #['a','c','ell_N','ell_L','n_N','n_L'] # outputs
        self.intertemps_hh = ['vbeg_a'] # intertemporal variables
  
        # c. GE
        self.shocks = [] # exogenous shocks
        self.unknowns = [] # endogenous unknowns
        self.targets = [] # targets = 0

        # d. all variables
        self.varlist = [
            'A',
            'B',
            'C',
            'clearing_A',
            'clearing_C',
            'clearing_N',
            'd_N',
            'd_L',
            'G',
            'i',
            'N_N',
            'N_L',
            'N',
            'ell_N',
            'ell_L',
            'M',
            'M_N',
            'M_L',
            'NKPC_res_N',
            'NKPC_res_L',
            'pi',
            'adjcost_N',
            'adjcost_L',
            'adjcost',
            'r',
            'istar',
            'tau',
            'w_L',
            'w_N',
            'w',
            'mc_N',
            'mc_L',
            'Y_N',
            'Y_L',
            'Z'] 

        # e. functions
        self.solve_hh_backwards = household_problem.solve_hh_backwards
        self.block_pre = blocks.block_pre
        self.block_post = blocks.block_post

    def setup(self):
        """ set baseline parameters """

        par = self.par

        par.Nfix = 2 # number of fixed discrete states (either work in L or N sector)
        par.Nz = 7 # number of stochastic discrete states (here productivity)

        # a. preferences
        par.beta = 0.98 # discount factor (guess, calibrated in ss)
        par.varphi = 0.80 # disutility of labor (guess, calibrated in ss)

        par.sigma = 2.0 # inverse of intertemporal elasticity of substitution
        par.nu = 2.0 # inverse Frisch elasticity
        #par.beta = 0.96 # discount factor
        #par.sigma = 2.0 # CRRA coefficient
        #par.nu = 0.0 # inverse Frisch elasticity
        #par.varphi = 0.0 # disutility of labor scaling?
        #par.gamma_hh = 0.0 # consumption substitution elasticity 
        #par.alpha_hh = 0.0 # consumption cobb-douglas loading


        # b. income parameters
        par.rho_z        = 0.96                                 # AR(1) parameter
        par.sigma_psi    = np.sqrt(0.50**2*(1-par.rho_z**2))    # std. of shock
        par.mu_psi       = 0.0                                  # mean of shock
        par.tau          = 0.0                                  # tax

        # c. production
        #par.alpha        = 0.36
        #par.gamma        = 0.8                                  # Elasticity of substitution
        #par.mu           = 1.2                                  # mark-up
        #par.kappa        = 0.1                                  # slope of Phillips curve
        #par.Gamma_ss     = 1.0
        #par.pm           = 1.0

        par.alpha_L      = 0.36                                 # cobb-douglas for sector L
        par.alpha_N      = 0.27                                 # cobb-douglas for sector N
        par.gamma_L      = 0.8                                  # substitution elasticity for sector L
        par.gamma_N      = 1.02                                 # substitution elasticity for sector N
        par.mu_L         = 1.4                                  # mark-up for sector L
        par.mu_N         = 1.1                                  # mark-up for sector N
        par.kappa_L      = 0.1                                  # price rigidity for sector L
        par.kappa_N      = 0.2                                  # price rigidity for sector N
        par.Gamma_ss     = 1.0                                  # direct approach: technology level in steady state
        par.pm           = 1.0 

        # d. government
        par.phi = 1.5 # Taylor rule coefficient on inflation
        par.phi_y = 0.0 # Taylor rule coefficient on output

        par.G_target_ss = 0.0 # government spending
        par.B_target_ss = 5.6 # bond supply

        # f. grids         
        par.a_min = 0.0 # minimum point in grid for a
        par.a_max = 100.0 # maximum point in grid for a
        par.Na = 500 # number of grid points

        # g. indirect approach: targets for stationary equilibrium
        par.r_target_ss = 0.005
        #par.r_ss_target = 0.03
        #par.w_ss_target = 1.0

        # h. misc.
        par.T = 500 # length of path        
        
        par.max_iter_solve = 50_000 # maximum number of iterations when solving
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating
        par.max_iter_broyden = 100 # maximum number of iteration when solving eq. system
        
        par.tol_ss = 1e-11 # tolerance when finding steady state
        par.tol_solve = 1e-11 # tolerance when solving
        par.tol_simulate = 1e-11 # tolerance when simulating
        par.tol_broyden = 1e-10 # tolerance when solving eq. system
        
        
    def allocate(self):
        """ allocate model """

        par = self.par

        self.allocate_GE() # should always be called here

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss