import numpy as np
import numba as nb

from GEModelTools import lag, lead
   
@nb.njit
def block_pre(par,ini,ss,path,ncols=1):

    for ncol in nb.prange(ncols):

        adjcost = path.adjcost[ncol,:]
        A_hh = path.A_hh[ncol,:]
        A = path.A[ncol,:]
        B = path.B[ncol,:]
        C_hh = path.C_hh[ncol,:]
        C = path.C[ncol,:]
        clearing_A = path.clearing_A[ncol,:]
        clearing_C = path.clearing_C[ncol,:]
        clearing_N = path.clearing_N[ncol,:]
        d = path.d[ncol,:]
        G = path.G[ncol,:]
        i = path.i[ncol,:]
        N_hh = path.N_hh[ncol,:]
        N = path.N[ncol,:]
        M = path.M[ncol,:]
        NKPC_res = path.NKPC_res[ncol,:]
        pi = path.pi[ncol,:]
        r = path.r[ncol,:]
        istar = path.istar[ncol,:]
        tau = path.tau[ncol,:]
        w = path.w[ncol,:]
        mc = path.mc[ncol,:]
        Y = path.Y[ncol,:]
        Z = path.Z[ncol,:]

        #################
        # implied paths #
        #################

        # a. firms
        mc[:] = ((1-par.alpha)*(w*Z)**(1-par.gamma)+par.alpha*par.pm**(1-par.gamma))**(1/(1-par.gamma))
        N[:] = (w/mc)**(-par.gamma)*(1-par.alpha)*Z**(1-par.gamma)*Y
        M[:] = (par.pm/mc)**(-par.gamma)*par.alpha*Y
        adjcost[:] = par.mu/(par.mu-1)/(2*par.kappa)*np.log(1+pi)**2*Y
        d[:] = Y-w*N-par.pm*M-adjcost

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
        C[:] = Y-G-adjcost-par.pm*M

@nb.njit
def block_post(par,ini,ss,path,ncols=1):

    for ncol in nb.prange(ncols):

        adjcost = path.adjcost[ncol,:]
        A_hh = path.A_hh[ncol,:]
        A = path.A[ncol,:]
        B = path.B[ncol,:]
        C_hh = path.C_hh[ncol,:]
        C = path.C[ncol,:]
        clearing_A = path.clearing_A[ncol,:]
        clearing_C = path.clearing_C[ncol,:]
        clearing_N = path.clearing_N[ncol,:]
        d = path.d[ncol,:]
        G = path.G[ncol,:]
        i = path.i[ncol,:]
        N_hh = path.N_hh[ncol,:]
        N = path.N[ncol,:]
        M = path.M[ncol,:]
        NKPC_res = path.NKPC_res[ncol,:]
        pi = path.pi[ncol,:]
        r = path.r[ncol,:]
        istar = path.istar[ncol,:]
        tau = path.tau[ncol,:]
        w = path.w[ncol,:]
        mc = path.mc[ncol,:]
        Y = path.Y[ncol,:]
        Z = path.Z[ncol,:]

        #################
        # check targets #
        #################

        # a. phillips curve
        r_plus = lead(r,ss.r)
        pi_plus = lead(pi,ss.pi)
        Y_plus = lead(Y,ss.Y)

        mc[:] = ((1-par.alpha)*(w*Z)**(1-par.gamma)+par.alpha*par.pm**(1-par.gamma))**(1/(1-par.gamma))
        mc[:] = ((1-par.alpha)*(w*Z)**(1-par.gamma)+par.alpha*par.pm**(1-par.gamma))**(1/(1-par.gamma))
        
        NKPC_res[:] = par.kappa*(mc-1/par.mu) + Y_plus/Y*np.log(1+pi_plus)/(1+r_plus) - np.log(1+pi)


        # b. market clearing
        clearing_A[:] = A-A_hh
        clearing_C[:] = C-C_hh
        clearing_N[:] = N-N_hh