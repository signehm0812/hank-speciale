
import numpy as np
from EconModel import EconModelClass
from GEModelTools import GEModelClass

import household_problem_baseline
import steady_state_baseline
import blocks_baseline

class HANKModelClass_baseline(EconModelClass,GEModelClass):
    
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
        self.inputs_hh = ['w','r','i','d','tau'] # direct inputs
        self.inputs_hh_z = [] # transition matrix inputs
        self.outputs_hh = ['a','c','ell','n','u'] # outputs
        self.intertemps_hh = ['vbeg_a'] # intertemporal variables

        # c. GE
        self.shocks = ['istar','Z','pm'] # exogenous inputs
        self.unknowns = ['pi','w','Y'] # endogenous inputs
        self.targets = ['NKPC_res','clearing_A','clearing_N'] # targets
        
        # d. all variables
        self.varlist = [ # all variables
            'A',
            'B',
            'C',
            'clearing_A',
            'clearing_C',
            'clearing_N',
            'd',
            'G',
            'i',
            'N',
            'M',
            'NKPC_res',
            'pi',
            'P',
            'pm',
            'adjcost',
            'r',
            'istar',
            'tau',
            'w',
            'mc',
            'Y',
            'Y_star',
            'Z']

        # e. functions
        self.solve_hh_backwards = household_problem_baseline.solve_hh_backwards
        self.block_pre = blocks_baseline.block_pre
        self.block_post = blocks_baseline.block_post
        
    def setup(self):
        """ set baseline parameters """

        par = self.par

        par.Nfix               = 1                     # number of fixed discrete states (either work in L or N sector)
        par.Nz                 = 7                     # number of stochastic discrete states (here productivity)
        par.r_target_ss        = 1.02**(1/4)-1

        # a. preferences
        par.beta               = 0.9875                # discount factor (guess, calibrated in ss)
        par.varphi             = 1.0                   # disutility of labor
        par.sigma              = 2.0                   # inverse of intertemporal elasticity of substitution
        par.nu                 = 2.0                   # inverse Frisch elasticity

        # c. income parameters     
        par.rho_z              = 0.9777                # AR(1) parameter
        par.sigma_psi          = 0.1928                # std. of psi

        #par.rho_z = 0.966                                     
        #par.sigma_psi = np.sqrt(0.50**2*(1-par.rho_z**2))      
    
        # d. price setting           
        par.alpha            = 0.5                    # cobb-douglas
        par.gamma            = 0.5                    # substitution elasticity
        par.mu               = 1.5                     # mark-up 
        par.kappa            = 0.125                    # price rigidity 
    
        # e. government
        par.phi                = 1.5                    # Taylor rule coefficient on inflation
        par.phi_y              = 0.0                    # Taylor rule coefficient on output
              
        par.G_target_ss        = 0.0                    # government spending
        par.B_target_ss        = 5.6                    # bond supply
     
        # f. grids               
        par.a_min              = 0.0                    # maximum point in grid for a
        par.a_max              = 100.0                  # maximum point in grid for a
        par.Na                 = 500                    # number of grid points
     
        # g. shocks      
        par.jump_Z             = 0.0                    # initial jump
        par.rho_Z              = 0.00                   # AR(1) coefficeint
        par.std_Z              = 0.00                   # std.
     
        #par.jump_istar =       0.0025
        par.jump_istar         = 0.0
        par.rho_istar          = 0.61
        par.std_istar          = 0.0025
     
        #par.jump_pm            = 0.01
        par.jump_pm            = 0.0
        par.rho_pm             = 0.9
        par.std_pm             = 0.01

        # h. misc.
        par.T                  = 500                    # length of path        
        par.max_iter_solve     = 50_000                 # maximum number of iterations when solving
        par.max_iter_simulate  = 50_000                 # maximum number of iterations when simulating
        par.max_iter_broyden   = 100                    # maximum number of iteration when solving eq. system
        par.tol_ss             = 1e-11                  # tolerance when finding steady state
        par.tol_solve          = 1e-11                  # tolerance when solving
        par.tol_simulate       = 1e-11                  # tolerance when simulating
        par.tol_broyden        = 1e-10                  # tolerance when solving eq. system

        
    def allocate(self):
        """ allocate model """

        par = self.par

        # b. solution
        self.allocate_GE()

    prepare_hh_ss = steady_state_baseline.prepare_hh_ss
    find_ss = steady_state_baseline.find_ss        