
import numpy as np
from EconModel import EconModelClass
from GEModelTools import GEModelClass

import household_problem
import steady_state
import blocks

class HANKModelClass(EconModelClass,GEModelClass):
    
    #########
    # setup #
    #########      

    def settings(self):
        """ fundamental settings """

        # a. namespaces
        self.namespaces = ['par','ss','ini','path','sim']
        
        # b. household
        self.grids_hh = ['a'] # grids
        self.pols_hh = ['a'] # policy functions
        self.inputs_hh = ['w_N','w_L','r','d_N','d_L','tau'] # direct inputs
        self.inputs_hh_z = [] # transition matrix inputs
        self.outputs_hh = ['a','c','ell','n'] # outputs
        self.intertemps_hh = ['vbeg_a'] # intertemporal variables

        # c. GE
        self.shocks = ['istar','Z','pm'] # exogenous inputs
        self.unknowns = ['pi_L','pi_N','w_L','w_N','Y_L','Y_N'] # endogenous inputs
        self.targets = ['NKPC_res_N','NKPC_res_L','clearing_A','clearing_N'] # targets
        
        # d. all variables
        self.varlist = [ # all variables
            'A',
            'B',
            'C',
            'clearing_A',
            'clearing_C',
            'clearing_N',
            'd',
            'd_N',
            'd_L',
            'G',
            'i',
            'N',
            'M',
            'M_N',
            'M_L',
            'pm',
            #'NKPC_res',
            'NKPC_res_N',
            'NKPC_res_L',
            'pi',
            'pi_N',
            'pi_L',                        
            'adjcost',
            'adjcost_N',
            'adjcost_L',            
            'r',
            'istar',
            'tau',
            'w',
            'w_N',
            'w_L',            
            'mc_N',
            'mc_L',
            'Y',
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

        par.Nfix                = 1                                   # number of fixed discrete states (either work in L or N sector)
        par.Nz                  = 7                                   # number of stochastic discrete states (here productivity)
        par.r_target_ss         = 0.005

        # a. preferences
        par.beta = 0.9875 # discount factor (guess, calibrated in ss)
        par.varphi = 0.8 # disutility of labor (guess, calibrated in ss)

        par.sigma = 2.0 # inverse of intertemporal elasticity of substitution
        par.nu = 2.0 # inverse Frisch elasticity
        
        # c. income parameters
        par.rho_z = 0.966 # AR(1) parameter
        par.sigma_psi = np.sqrt(0.50**2*(1-par.rho_z**2)) # std. of psi

        # d. price setting
        par.alpha_L             = 0.3                                 # cobb-douglas for sector L
        par.alpha_N             = 0.4                                 # cobb-douglas for sector N
        par.gamma_L             = 1.1                                 # substitution elasticity for sector L
        par.gamma_N             = 1.2                                 # substitution elasticity for sector N
        par.mu_L                = 1.1                                 # mark-up for sector L
        par.mu_N                = 1.15                                 # mark-up for sector N
        par.kappa_L             = 0.1                                 # price rigidity for sector L
        par.kappa_N             = 0.15                                 # price rigidity for sector N
        #par.Gamma_ss           = 1.0                                 # direct approach: technology level in steady state

        #par.M_N                 = 1.2
        #par.M_L                 = 0.8
        #par.M = par.M_L + par.M_N

        #par.alpha               = 0.3
        #par.gamma               = 1.1                                 # Elasticity of substitution
        #par.mu                  = 1.2                                 # mark-up
        #par.kappa               = 0.1                                 # slope of Phillips curve
        #par.pm                  = 0.8
        #par.M                   = 1.2

        # e. government
        par.phi = 1.5 # Taylor rule coefficient on inflation
        par.phi_y = 0.0 # Taylor rule coefficient on output
        
        par.G_target_ss = 0.0 # government spending
        par.B_target_ss = 5.6 # bond supply

        # f. grids         
        par.a_min = 0.0 # maximum point in grid for a
        par.a_max = 100.0 # maximum point in grid for a
        par.Na = 500 # number of grid points

        # g. shocks
        par.jump_Z      = 0.0 # initial jump
        par.rho_Z       = 0.00 # AR(1) coefficeint
        par.std_Z       = 0.00 # std.
        par.jump_istar  = -0.0025
        par.rho_istar   = 0.61
        par.std_istar   = 0.0025
        par.jump_pm     = 0.25
        par.rho_pm      = 0.7
        par.std_pm      = 0.0025

        # h. misc.
        par.T = 1000 # length of path        
        
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

        # b. solution
        self.allocate_GE()

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss        