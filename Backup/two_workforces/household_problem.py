# solving the household problem

import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec

@nb.njit        
def solve_hh_backwards(par,z_trans,w_N,w_L,r,d_N,d_L,tau,vbeg_a_plus,vbeg_a,a,c,ell_N,ell_L,n_N,n_L,n):
    """ solve backwards with vbeg_a_plus from previous iteration """

    for i_fix in range(par.Nfix):
        
        # a. solution step
        for i_z in range(par.Nz):

            # i. prepare
            z = par.z_grid[i_z]
            T = (d_L+d_N)*z - tau*z
            fac_N = (w_N*z/par.varphi)**(1/par.nu)
            fac_L = (w_L*z/par.varphi)**(1/par.nu)

            # ii. use focs
            c_endo = (par.beta*vbeg_a_plus[i_fix,i_z,:])**(-1/par.sigma)
            ell_endo_N = fac_N*(par.beta*vbeg_a_plus[i_fix,i_z,:])**(1/par.nu)
            ell_endo_L = fac_L*(par.beta*vbeg_a_plus[i_fix,i_z,:])**(1/par.nu)
            n_endo_N = ell_endo_N*z
            n_endo_L = ell_endo_L*z

            # iii. re-interpolate
            m_endo = c_endo + par.a_grid - (w_N*n_endo_N+w_L*n_endo_L) - T
            m_exo = (1+r)*par.a_grid

            interp_1d_vec(m_endo,c_endo,m_exo,c[i_fix,i_z,:])
            interp_1d_vec(m_endo,ell_endo_N,m_exo,ell_N[i_fix,i_z,:])
            interp_1d_vec(m_endo,ell_endo_L,m_exo,ell_L[i_fix,i_z,:])
            n_N[i_fix,i_z,:] = ell_N[i_fix,i_z,:]*z
            n_L[i_fix,i_z,:] = ell_L[i_fix,i_z,:]*z

            # iv. saving
            a[i_fix,i_z,:] = m_exo + (w_N*n_N[i_fix,i_z,:]+w_L*n_L[i_fix,i_z,:]) + T - c[i_fix,i_z,:]

            # v. refinement at constraint
            for i_a in range(par.Na):

                if a[i_fix,i_z,i_a] < par.a_min:
                    
                    # i. binding constraint for a
                    a[i_fix,i_z,i_a] = par.a_min

                    # ii. solve foc for n
                    elli_N = ell_N[i_fix,i_z,i_a]
                    elli_L = ell_L[i_fix,i_z,i_a]
                    for i in range(60):

                        ci = (1+r)*par.a_grid[i_a] + (w_N*elli_N+w_L*elli_L)*z + T - par.a_min # from binding constraint

                        error_N = elli_N - fac_N*ci**(-par.sigma/par.nu)
                        error_L = elli_L - fac_L*ci**(-par.sigma/par.nu)
                        if (np.abs(error_N) < 1e-11 and np.abs(error_L) < 1e-11):
                            break
                        else:
                            derror_N = 1 - fac_N*(-par.sigma/par.nu)*ci**(-par.sigma/par.nu-1)*w_N*z
                            derror_L = 1 - fac_L*(-par.sigma/par.nu)*ci**(-par.sigma/par.nu-1)*w_L*z
                            elli_N = elli_N - error_N/derror_N
                            elli_L = elli_L - error_L/derror_L
                    else:
                        
                        raise ValueError('solution could not be found')

                    # iii. save
                    c[i_fix,i_z,i_a] = ci
                    ell_N[i_fix,i_z,i_a] = elli_N
                    ell_L[i_fix,i_z,i_a] = elli_L
                    n_N[i_fix,i_z,i_a] = elli_N*z
                    n_L[i_fix,i_z,i_a] = elli_L*z
                    n[i_fix,i_z,i_a] = n_N[i_fix,i_z,i_a] + n_L[i_fix,i_z,i_a]
                    
                else:

                    break

        # b. expectation step
        v_a = c[i_fix,:,:]**(-par.sigma)
        vbeg_a[i_fix] = (1+r)*z_trans[i_fix]@v_a