# solving the household problem

import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec

@nb.njit        
def solve_hh_backwards(par,z_trans,w,r,i,d,tau,vbeg_a_plus,vbeg_a,a,c,ell,n,u):
    """ solve backwards with vbeg_a_plus from previous iteration """

    for i_fix in range(par.Nfix):
        
        # a. solution step
        for i_z in range(par.Nz):

            # i. prepare
            z = par.z_grid[i_z]
            T = d*z - tau*z + par.chi 
            fac = (w*z/par.varphi)**(1/par.nu)

            # ii. use focs
            c_endo = (par.beta*vbeg_a_plus[i_fix,i_z,:])**(-1/par.sigma)
            ell_endo = fac*(par.beta*vbeg_a_plus[i_fix,i_z,:])**(1/par.nu)
            n_endo = ell_endo*z

            # iii. re-interpolate
            m_endo = c_endo + par.a_grid - w*n_endo - T + par.c_bar
            m_exo = (1+r)*par.a_grid

            interp_1d_vec(m_endo,c_endo,m_exo,c[i_fix,i_z,:])
            interp_1d_vec(m_endo,ell_endo,m_exo,ell[i_fix,i_z,:])
            n[i_fix,i_z,:] = ell[i_fix,i_z,:]*z

            # iv. saving
            a[i_fix,i_z,:] = m_exo + w*n[i_fix,i_z,:] + T - c[i_fix,i_z,:] - par.c_bar

            # v. refinement at constraint
            for i_a in range(par.Na):

                if a[i_fix,i_z,i_a] < par.a_min:
                    
                    # i. binding constraint for a
                    a[i_fix,i_z,i_a] = par.a_min

                    # ii. solve foc for n
                    elli = ell[i_fix,i_z,i_a]
                    for i in range(30):

                        ci = (1+r)*par.a_grid[i_a] + w*z*elli + T - par.a_min - par.c_bar# from binding constraint

                        error = elli - fac*ci**(-par.sigma/par.nu)
                        if np.abs(error) < 1e-11:
                            break
                        else:
                            derror = 1 - fac*(-par.sigma/par.nu)*ci**(-par.sigma/par.nu-1)*w*z
                            elli = elli - error/derror
                    else:
                        
                        raise ValueError('solution could not be found')

                    # iii. save
                    c[i_fix,i_z,i_a] = ci
                    ell[i_fix,i_z,i_a] = elli
                    n[i_fix,i_z,i_a] = elli*z
                    
                    
                else:

                    break
                            
        # b. expectation step
        u[i_fix,:,:] = (c[i_fix,:,:])**(1-par.sigma)/(1-par.sigma)-par.varphi*((ell[i_fix,:,:])**(1+par.nu))/(1+par.nu)
        #u[i_fix,:,:] = ci**(1-par.sigma)/(1-par.sigma)-par.varphi*(elli**(1+par.nu))/(1+par.nu)
        v_a = c[i_fix,:,:]**(-par.sigma)
        vbeg_a[i_fix] = (1+r)*z_trans[i_fix]@v_a
        c[i_fix,:,:] = c[i_fix,:,:] + par.c_bar
