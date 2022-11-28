import numpy as np
import numba as nb

from GEModelTools import lag, lead
   
@nb.njit
def block_pre(par,ini,ss,path,ncols=1):

    for ncol in nb.prange(ncols):

        A = path.A[ncol,:]
        B = path.B[ncol,:]
        C = path.C[ncol,:]
        C_N = path.C_N[ncol,:]
        C_L = path.C_L[ncol,:]
        clearing_A = path.clearing_A[ncol,:]
        clearing_C = path.clearing_C[ncol,:]
        clearing_C_N = path.clearing_C_N[ncol,:]
        clearing_C_L = path.clearing_C_L[ncol,:]
        clearing_N = path.clearing_N[ncol,:]
        d = path.d[ncol,:]
        d_N = path.d_N[ncol,:]
        d_L = path.d_L[ncol,:]
        G = path.G[ncol,:]
        i = path.i[ncol,:]
        N = path.N[ncol,:]
        N_N = path.N_N[ncol,:]
        N_L = path.N_L[ncol,:]
        M_N = path.M_N[ncol,:]
        M_L = path.M_L[ncol,:]
        pm_L = path.pm_L[ncol,:]
        pm_N = path.pm_N[ncol,:]
        NKPC_res_N = path.NKPC_res_N[ncol,:]
        NKPC_res_L = path.NKPC_res_L[ncol,:]
        pi = path.pi[ncol,:]
        pi_N = path.pi_N[ncol,:]
        pi_L = path.pi_L[ncol,:]
        adjcost = path.adjcost[ncol,:]
        adjcost_N = path.adjcost_N[ncol,:]
        adjcost_L = path.adjcost_L[ncol,:]
        r = path.r[ncol,:]
        istar = path.istar[ncol,:]
        rstar = path.rstar[ncol,:]
        tau = path.tau[ncol,:]
        w = path.w[ncol,:]
        w_N = path.w_N[ncol,:]
        w_L = path.w_L[ncol,:]
        mc_N = path.mc_N[ncol,:]
        mc_L = path.mc_L[ncol,:]
        Y = path.Y[ncol,:]
        Y_N = path.Y_N[ncol,:]
        Y_L = path.Y_L[ncol,:]
        Y_star = path.Y_star[ncol,:]
        Q = path.Q[ncol,:]
        Q_check = path.Q_check[ncol,:]
        p_N = path.p_N[ncol,:]
        p_L = path.p_L[ncol,:]
        P = path.P[ncol,:]
        Z_N = path.Z_N[ncol,:]
        Z_L = path.Z_L[ncol,:]
        A_hh = path.A_hh[ncol,:]
        C_hh = path.C_hh[ncol,:]
        C_HAT_N_hh = path.C_HAT_N_hh[ncol,:]
        C_N_hh = path.C_N_hh[ncol,:]
        C_L_hh = path.C_L_hh[ncol,:]
        ELL_hh = path.ELL_hh[ncol,:]
        N_hh = path.N_hh[ncol,:]
        P_hh = path.P_hh[ncol,:]

        #################
        # implied paths #
        #################

        # inflation
        #Q_lag = lag(ini.Q,Q)
        pi_L[:] = (Q/ss.Q)*(1+pi_N)-1
        #pi_L[:] = (Q/Q_lag)*(1+pi_N)-1
        
        # back out prices option 1 - inflation is multiplicative, standard solution goes wrong after 5 periods?

        #p_N_lag = lag(ini.p_N,p_N)
        #p_N[:] = (pi_N+1)*p_N_lag
        #p_L[:] = Q*p_N


        # back out prices option 2 - inflation is multiplicative, and needs to be looped?              
        #for t in range(par.T):
        #
        #    # i. lag
        #    p_N_lag = p_N[t-1] if t > 0 else ss.p_N
        #    p_N[t] = (pi_N[t]+1)*p_N_lag
        #    p_L[t] = Q[t]*p_N[t]
        
                
        # back out prices option 3 - inflation is just relative to steady state         
        p_N[:] = (1+pi_N)*ss.p_N
        p_L[:] = (1+pi_L)*ss.p_L # - What goes wrong here? Model still runs fine, but not as it should
        #p_L[:] = Q*p_N
        Q_check[:] = p_L/p_N
        
        #Inflation option one
        #P_hh_lag = lag(ini.P_hh,P_hh)
        #pi[:] = P_hh/P_hh_lag-1 #preliminary inflation indexing - only works if inflation is as above
        
        #Inflation option two - standard        
        pi[:] = (1+pi_N)**par.epsilon*(1+pi_L)**(1-par.epsilon)-1 #preliminary inflation indexing
        
        # prices
        P[:] = (par.alpha_hh*p_N**(1-par.gamma_hh)+(1-par.alpha_hh)*p_L**(1-par.gamma_hh))**(1/(1-par.gamma_hh)) # price index
        #P[:] = ((1+pi_N)**(1-par.gamma_hh)*par.alpha_hh+(1-par.alpha_hh)*Q**(1-par.gamma_hh))**(1/(1-par.gamma_hh)) # price index
        w_L[:] = (1/Q)*w_N # real wage rate
        pm_L[:] = (1/Q)*pm_N # real raw material price

        # production
        mc_N[:] = ((1-par.alpha_N)*(w_N/Z_N)**(1-par.gamma_N)+par.alpha_N*pm_N**(1-par.gamma_N))**(1/(1-par.gamma_N)) # marginal cost sector N
        mc_L[:] = ((1-par.alpha_L)*(w_L/Z_L)**(1-par.gamma_L)+par.alpha_L*pm_L**(1-par.gamma_L))**(1/(1-par.gamma_L)) # marginal cost sector L

        Y_N[:] = N_N/((1-par.alpha_N)*(w_N/mc_N)**(-par.gamma_N)*Z_N**(par.gamma_N-1))
        Y_L[:] = N_L/((1-par.alpha_L)*(w_L/mc_L)**(-par.gamma_L)*Z_L**(par.gamma_L-1))

        adjcost_N[:] = Y_N*(par.mu_N/(par.mu_N-1))*(1/(2*par.kappa_N))*(np.log(1+pi_N))**2 # adjustment cost sector N
        adjcost_L[:] = Y_L*(par.mu_L/(par.mu_L-1))*(1/(2*par.kappa_L))*(np.log(1+pi_L))**2 # adjustment cost sector L

        Y[:] = (Y_N+Q*Y_L)*(p_N/P) # overall production
        Y_star[:] = (ss.Y_N+Q*ss.Y_L)*(p_N/P) # potential production

        M_N[:] = par.alpha_N*(pm_N/mc_N)**(-par.gamma_N)*Y_N # M_N demand
        M_L[:] = par.alpha_L*(pm_L/mc_L)**(-par.gamma_L)*Y_L # M_L demand

        d_N[:] = Y_N-w_N*N_N-pm_N*M_N-adjcost_N # dividends sector N
        d_L[:] = Y_L-w_L*N_L-pm_L*M_L-adjcost_L # dividends sector L

        # b. monetary policy
        i[:] = istar + par.phi*pi + par.phi_y*(Y-(Y_star)) # taylor rule
        i_lag = lag(ini.i,i)
        r[:] = (1+i_lag)/(1+pi)-1 ## Fix these taylor rule weights 
        #r[:] = i-pi # fisher equation

        # c. government
        B[:] = ss.B
        tau[:] = r*B + par.chi
        G[:] = tau-r*B - par.chi
        
        # d. aggregates
        A[:] = ss.B
        C_N[:] = Y_N-adjcost_N-pm_N*M_N
        C_L[:] = Y_L-adjcost_L-pm_L*M_L
        C[:] = (C_N + Q*C_L)*(p_N/P)
        N[:] = N_N + N_L

@nb.njit
def block_post(par,ini,ss,path,ncols=1):

    for ncol in nb.prange(ncols):

        A = path.A[ncol,:]
        B = path.B[ncol,:]
        C = path.C[ncol,:]
        C_N = path.C_N[ncol,:]
        C_L = path.C_L[ncol,:]
        clearing_A = path.clearing_A[ncol,:]
        clearing_C = path.clearing_C[ncol,:]
        clearing_C_N = path.clearing_C_N[ncol,:]
        clearing_C_L = path.clearing_C_L[ncol,:]
        clearing_N = path.clearing_N[ncol,:]
        d = path.d[ncol,:]
        d_N = path.d_N[ncol,:]
        d_L = path.d_L[ncol,:]
        G = path.G[ncol,:]
        i = path.i[ncol,:]
        N = path.N[ncol,:]
        N_N = path.N_N[ncol,:]
        N_L = path.N_L[ncol,:]
        M_N = path.M_N[ncol,:]
        M_L = path.M_L[ncol,:]
        pm_L = path.pm_L[ncol,:]
        pm_N = path.pm_N[ncol,:]
        NKPC_res_N = path.NKPC_res_N[ncol,:]
        NKPC_res_L = path.NKPC_res_L[ncol,:]
        pi = path.pi[ncol,:]
        pi_N = path.pi_N[ncol,:]
        pi_L = path.pi_L[ncol,:]
        adjcost = path.adjcost[ncol,:]
        adjcost_N = path.adjcost_N[ncol,:]
        adjcost_L = path.adjcost_L[ncol,:]
        r = path.r[ncol,:]
        istar = path.istar[ncol,:]
        rstar = path.rstar[ncol,:]
        tau = path.tau[ncol,:]
        w = path.w[ncol,:]
        w_N = path.w_N[ncol,:]
        w_L = path.w_L[ncol,:]
        mc_N = path.mc_N[ncol,:]
        mc_L = path.mc_L[ncol,:]
        Y = path.Y[ncol,:]
        Y_N = path.Y_N[ncol,:]
        Y_L = path.Y_L[ncol,:]
        Y_star = path.Y_star[ncol,:]
        Q = path.Q[ncol,:]
        P = path.P[ncol,:]
        Z_N = path.Z_N[ncol,:]
        Z_L = path.Z_L[ncol,:]
        A_hh = path.A_hh[ncol,:]
        C_hh = path.C_hh[ncol,:]
        C_HAT_N_hh = path.C_HAT_N_hh[ncol,:]
        C_N_hh = path.C_N_hh[ncol,:]
        C_L_hh = path.C_L_hh[ncol,:]
        ELL_hh = path.ELL_hh[ncol,:]
        N_hh = path.N_hh[ncol,:]
        p_N = path.p_N[ncol,:]
        P_hh = path.P_hh[ncol,:]

        #################
        # check targets #
        #################

        # a. phillips curve
        r_plus = lead(r,ss.r)
        pi_N_plus = lead(pi_N,ss.pi_N)
        pi_L_plus = lead(pi_L,ss.pi_L)
        Y_N_plus = lead(Y_N,ss.Y_N)
        Y_L_plus = lead(Y_L,ss.Y_L)

        NKPC_res_N[:] = par.kappa_N*(mc_N-1/par.mu_N) + Y_N_plus/Y_N*np.log(1+pi_N_plus)/(1+r_plus) - np.log(1+pi_N)
        NKPC_res_L[:] = par.kappa_L*(mc_L-1/par.mu_L) + Y_L_plus/Y_L*np.log(1+pi_L_plus)/(1+r_plus) - np.log(1+pi_L)

        # b. market clearing
        clearing_A[:] = A-A_hh
        clearing_C_N[:] = C_N-C_N_hh
        clearing_C_L[:] = C_L-C_L_hh
        clearing_N[:] = N-N_hh
        clearing_C[:] = C-C_hh