import numpy as np
import numba as nb

from GEModelTools import lag, lead
   
@nb.njit
def block_pre(par,ini,ss,path,ncols=1):

    for ncol in nb.prange(ncols):

        adjcost = path.adjcost[ncol,:]
        adjcost_N = path.adjcost_N[ncol,:]
        adjcost_L = path.adjcost_L[ncol,:]
        A_hh = path.A_hh[ncol,:]
        A = path.A[ncol,:]
        B = path.B[ncol,:]
        C_hh = path.C_hh[ncol,:]
        C = path.C[ncol,:]
        clearing_A = path.clearing_A[ncol,:]
        clearing_C = path.clearing_C[ncol,:]
        clearing_N = path.clearing_N[ncol,:]
        d_N = path.d_N[ncol,:]
        d_L = path.d_L[ncol,:]
        G = path.G[ncol,:]
        i = path.i[ncol,:]
        N_hh = path.N_hh[ncol,:]
        N_N = path.N_N[ncol,:]
        N_L = path.N_L[ncol,:]
        M_N = path.M_N[ncol,:]
        M_L = path.M_L[ncol,:]
        NKPC_res_N = path.NKPC_res_N[ncol,:]
        NKPC_res_L = path.NKPC_res_L[ncol,:]
        pi = path.pi[ncol,:]
        r = path.r[ncol,:]
        istar = path.istar[ncol,:]
        tau = path.tau[ncol,:]
        w_N = path.w_N[ncol,:]
        w = path.w[ncol,:]
        w_L = path.w_L[ncol,:]
        mc_N = path.mc_N[ncol,:]
        mc_L = path.mc_L[ncol,:]
        Y_N = path.Y_N[ncol,:]
        Y = path.Y[ncol,:]
        Y_L = path.Y_L[ncol,:]
        Z = path.Z[ncol,:]

        #################
        # implied paths #
        #################

        # a. firms
        # sector N
        mc_N[:] = ((1-par.alpha_N)*(w_N*Z)**(1-par.gamma_N)+par.alpha_N*par.pm**(1-par.gamma_N))**(1/(1-par.gamma_N))
        N_N[:] = (w_N/mc_N)**(-par.gamma_N)*(1-par.alpha_N)*Z**(1-par.gamma_N)*Y_N
        M_N[:] = (par.pm/mc_N)**(-par.gamma_N)*par.alpha_N*Y_N
        adjcost_N[:] = par.mu_N/(par.mu_N-1)/(2*par.kappa_N)*np.log(1+pi)**2*Y_N
        d_N[:] = Y_N-w_N*N_N-par.pm*M_N-adjcost_N
        
        # sector L
        mc_L[:] = ((1-par.alpha_L)*(w_L*Z)**(1-par.gamma_L)+par.alpha_L*par.pm**(1-par.gamma_L))**(1/(1-par.gamma_L))
        N_L[:] = (w_L/mc_L)**(-par.gamma_L)*(1-par.alpha_L)*Z**(1-par.gamma_L)*Y_L
        M_L[:] = (par.pm/mc_L)**(-par.gamma_L)*par.alpha_L*Y_L
        adjcost_L[:] = par.mu_L/(par.mu_L-1)/(2*par.kappa_L)*np.log(1+pi)**2*Y_L
        d_L[:] = Y_L-w_L*N_L-par.pm*M_L-adjcost_L

        Y = Y_N+Y_L
        adjcost = adjcost_N+adjcost_L

        # b. monetary policy
        i[:] = istar + par.phi*pi + par.phi_y*(Y-ss.Y)
        i_lag = lag(ini.i,i)
        r[:] = (1+i_lag)/(1+pi)-1

        # c. government
        B[:] = ss.B
        tau[:] = r*B
        G[:] = tau-r*B
        
        # d. aggregates
        A[:] = B[:] = ss.B
        C[:] = Y-G-adjcost

@nb.njit
def block_post(par,ini,ss,path,ncols=1):

    for ncol in nb.prange(ncols):

        adjcost = path.adjcost[ncol,:]
        adjcost_N = path.adjcost_N[ncol,:]
        adjcost_L = path.adjcost_L[ncol,:]
        A_hh = path.A_hh[ncol,:]
        A = path.A[ncol,:]
        B = path.B[ncol,:]
        C_hh = path.C_hh[ncol,:]
        C = path.C[ncol,:]
        clearing_A = path.clearing_A[ncol,:]
        clearing_C = path.clearing_C[ncol,:]
        clearing_N = path.clearing_N[ncol,:]
        d_N = path.d_N[ncol,:]
        d_L = path.d_L[ncol,:]
        G = path.G[ncol,:]
        i = path.i[ncol,:]
        N_hh = path.N_hh[ncol,:]
        N_N = path.N_N[ncol,:]
        N_L = path.N_L[ncol,:]
        M_N = path.M_N[ncol,:]
        M_L = path.M_L[ncol,:]
        NKPC_res_N = path.NKPC_res_N[ncol,:]
        NKPC_res_L = path.NKPC_res_L[ncol,:]
        pi = path.pi[ncol,:]
        r = path.r[ncol,:]
        istar = path.istar[ncol,:]
        tau = path.tau[ncol,:]
        w_N = path.w_N[ncol,:]
        w_L = path.w_L[ncol,:]
        w = path.w[ncol,:]
        mc_N = path.mc_N[ncol,:]
        mc_L = path.mc_L[ncol,:]
        Y_N = path.Y_N[ncol,:]
        Y_L = path.Y_L[ncol,:]
        Y = path.Y[ncol,:]
        Z = path.Z[ncol,:]
        
        #################
        # check targets #
        #################

        # a. phillips curve
        r_plus = lead(r,ss.r)
        pi_plus = lead(pi,ss.pi)
        Y_N_plus = lead(Y_N,ss.Y_N)
        Y_L_plus = lead(Y_L,ss.Y_L)

        mc_L[:] = ((1-par.alpha_L)*(w_L*Z)**(1-par.gamma_L)+par.alpha_L*par.pm**(1-par.gamma_L))**(1/(1-par.gamma_L))
        mc_N[:] = ((1-par.alpha_N)*(w_N*Z)**(1-par.gamma_N)+par.alpha_N*par.pm**(1-par.gamma_N))**(1/(1-par.gamma_N))
        
        NKPC_res_N[:] = par.kappa_N*(mc_N-1/par.mu_N) + Y_N_plus/Y_N*np.log(1+pi_plus)/(1+r_plus) - np.log(1+pi)
        NKPC_res_L[:] = par.kappa_L*(mc_L-1/par.mu_L) + Y_L_plus/Y_L*np.log(1+pi_plus)/(1+r_plus) - np.log(1+pi)

        N = N_N+N_L

        # b. market clearing
        clearing_A[:] = A-A_hh
        clearing_C[:] = C-C_hh
        clearing_N[:] = N-N_hh