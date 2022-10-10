# solving the household problem

import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec

@nb.njit(parallel=True)        
def solve_hh_backwards(par,z_trans,tau,r,w,d,ell,vbeg_a_plus,vbeg_a,a,c): #
    """ solve backwards with vbeg_a from previous iteration (here vbeg_a_plus) """

    for i_fix in nb.prange(par.Nfix):

        # a. solve step
        for i_z in range(par.Nz):
            
            z = par.z_grid[i_z]
            T = d*z - tau*z 

            # i. EGM
            c_endo = (par.beta*vbeg_a_plus[i_fix,i_z])**(-1/par.sigma)
            m_endo = c_endo + par.a_grid - w*ell*z - T
            
            # ii. interpolation to fixed grid
            m = (1+r)*par.a_grid + w*ell*z
            interp_1d_vec(m_endo,par.a_grid,m,a[i_fix,i_z])
            a[i_fix,i_z,:] = np.fmax(a[i_fix,i_z,:],0.0) # enforce borrowing constraint
            c[i_fix,i_z] = m-a[i_fix,i_z] #skal noget tilføjes her??

        # b. expectation step
        v_a = (1+r)*c[i_fix,:,:]**(-par.sigma)
        vbeg_a[i_fix] = z_trans[i_fix]@v_a