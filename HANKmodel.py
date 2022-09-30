import numpy as np

from EconModel import EconModelClass
from GEModelTools import GEModelClass

#import steady_state
#import household_problem

class HANCModelClass(EconModelClass,GEModelClass):    

    def settings(self):
        """ fundamental settings """

        # a. namespaces (typically not changed)
        self.namespaces = ['par','ini','ss','path','sim'] # not used today: 'ini', 'path', 'sim'

        # not used today: .sim and .path
        
        # b. household
        self.grids_hh = ['a'] # grids
        self.pols_hh = ['a'] # policy functions
        self.inputs_hh = ['r','tau'] # direct inputs
        self.inputs_hh_z = [] # transition matrix inputs (not used)
        self.outputs_hh = ['a','c'] # outputs
        self.intertemps_hh = ['vbeg_a'] # intertemporal variables

        # c. GE
        self.shocks = [] # exogenous shocks
        self.unknowns = [] # endogenous unknowns
        self.targets = [] # targets = 0

        # d. all variables
        self.varlist = [
            'A_hh','C_hh',
            'C','Y','B',
            'clearing_B','clearing_C',
            'r','tau'] 

        # e. functions
        #self.solve_hh_backwards = household_problem.solve_hh_backwards
        self.block_pre = None # not used today
        self.block_post = None # not used today

    def setup(self):
        """ set baseline parameters """

        par = self.par

        par.Nfix = 2 # number of fixed discrete states (either work in L or N sector)
        par.Nz = 7 # number of stochastic discrete states (here productivity)

        # a. preferences
        par.beta = 0.96 # discount factor
        par.sigma = 2.0 # CRRA coefficient
        par.nu = 0.0
        par.varphi =0.0 
        par.gamma = 0.0
        par.alpha = 0.0


        # b. income parameters
        par.rho_z = 0.96 # AR(1) parameter
        par.sigma_psi = 0.10 # std. of shock
        par.mu_psi = 0.0
        par.tau = 0.0

        # c. production
        par.alpha_L = 0.36 # cobb-douglas for sector L
        par.alpha_N = 0.27 # cobb-douglas for sector N
        par.gamma_L = 0.6 # substitution elasticity for sector L
        par.gamma_N = 1.02 # substitution elasticity for sector N
        par.mu_L = 0.4 # mark-up for sector L
        par.mu_N = 0.2 # mark-up for sector N
        par.kappa_L = 0.1 # price rigidity for sector L
        par.kappa_N = 0.2 #price rigidity for sector N
        par.Gamma_ss = 1.0 # direct approach: technology level in steady state

        # f. grids         
        par.a_max = 100.0 # maximum point in grid for a
        par.Na = 500 # number of grid points

        # g. indirect approach: targets for stationary equilibrium
        par.r_ss_target = 0.03
        #par.w_ss_target = 1.0
        par.tau_ss_target = 0.3

        # h. misc.
        par.max_iter_solve = 50_000 # maximum number of iterations when solving household problem
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating household problem
        
        par.tol_solve = 1e-12 # tolerance when solving household problem
        par.tol_simulate = 1e-12 # tolerance when simulating household problem
        
    #def allocate(self):
    #    """ allocate model """
#
    #    par = self.par
#
    #    self.allocate_GE() # should always be called here
#
    #prepare_hh_ss = steady_state.prepare_hh_ss
    #find_ss = steady_state.find_ss