# solving the household problem

import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec

@nb.njit        
def solve_hh_backwards(par,z_trans,w_N,r,d_N,d_L,tau,Q,P,vbeg_a_plus,vbeg_a,a,c,c_hat_N,c_N,c_L,ell,n,p,u):
    """ solve backwards with vbeg_a_plus from previous iteration """

    for i_fix in range(par.Nfix):
        
        # a. solution step
        for i_z in range(par.Nz):

            # i. prepare
            z = par.z_grid[i_z]
            T = (1/P)*(d_N+Q*d_L)*z - tau*z + par.chi
            fac = ((1/P)*(w_N*z/par.varphi))**(1/par.nu)

            # ii. use focs
            c_endo = (par.beta*vbeg_a_plus[i_fix,i_z,:])**(-1/par.sigma)
            ell_endo = fac*(par.beta*vbeg_a_plus[i_fix,i_z,:])**(1/par.nu)
            n_endo = ell_endo*z

            # iii. re-interpolate
            m_endo = c_endo + par.a_grid - (w_N*n_endo-par.c_bar)/P - T
            m_exo = (1+r)*par.a_grid

            interp_1d_vec(m_endo,c_endo,m_exo,c[i_fix,i_z,:])
            interp_1d_vec(m_endo,ell_endo,m_exo,ell[i_fix,i_z,:])
            n[i_fix,i_z,:] = ell[i_fix,i_z,:]*z

            # iv. saving
            a[i_fix,i_z,:] = m_exo + (w_N*n[i_fix,i_z,:]-par.c_bar)/P + T - c[i_fix,i_z,:]

            # v. refinement at constraint
            for i_a in range(par.Na):

                if a[i_fix,i_z,i_a] < par.a_min:
                    
                    # i. binding constraint for a
                    a[i_fix,i_z,i_a] = par.a_min

                    # ii. solve foc for n
                    elli = ell[i_fix,i_z,i_a]
                    for i in range(30):

                        ci = (1+r)*par.a_grid[i_a] + (w_N*z*elli-par.c_bar)/P + T - par.a_min # from binding constraint

                        error = elli - fac*ci**(-par.sigma/par.nu)
                        if np.abs(error) < 1e-11:
                            break
                        else:
                            derror = 1 - fac*(-par.sigma/par.nu)*ci**(-par.sigma/par.nu-1)*(1/P)*w_N*z
                            elli = elli - error/derror
                    else:
                        
                        raise ValueError('solution could not be found')

                    # iii. save
                    c[i_fix,i_z,i_a] = ci 
                    ell[i_fix,i_z,i_a] = elli
                    n[i_fix,i_z,i_a] = elli*z

                else:

                    break

        c_hat_N[i_fix,:,:] = par.alpha_hh*P**(par.gamma_hh)*(c[i_fix,:,:]) #Correct (c without c_bar)
        c_N[i_fix,:,:] = c_hat_N[i_fix,:,:] + par.c_bar #Correct c_N 
        c_L[i_fix,:,:] = (P/Q)**(par.gamma_hh)*(1-par.alpha_hh)*(c[i_fix,:,:]) #Correct c_L 
        #p[i_fix,:,:] = (par.c_bar+P**par.gamma_hh*(par.alpha_hh*(c[i_fix,:,:]-par.c_bar)+(1-par.alpha_hh)*Q**(1-par.gamma_hh)*(c[i_fix,:,:]-par.c_bar)))/(c[i_fix,:,:])
        p[i_fix,:,:] = (par.c_bar+(P*(c[i_fix,:,:] - par.c_bar/P)))/(c[i_fix,:,:])
        u[i_fix,:,:] = (((par.alpha_hh**(1/par.gamma_hh)*(c_hat_N[i_fix,:,:])**((par.gamma_hh-1)/par.gamma_hh)+(1-par.alpha_hh)**(1/par.gamma_hh)*(c_L[i_fix,:,:])**((par.gamma_hh-1)/par.gamma_hh))**(par.gamma_hh/(par.gamma_hh-1)))**(1-par.sigma))/(1-par.sigma)-par.varphi*(((ell[i_fix,:,:])**(1+par.nu))/(1+par.nu))
        #u[i_fix,:,:] = ((c[i_fix,:,:])**(1-par.sigma))/(1-par.sigma)-par.varphi*(((ell[i_fix,:,:])**(1+par.nu))/(1+par.nu))

        # b. expectation step
        v_a = c[i_fix,:,:]**(-par.sigma)
        vbeg_a[i_fix] = (1+r)*z_trans[i_fix]@v_a
        c[i_fix,:,:] = c[i_fix,:,:] + par.c_bar/P