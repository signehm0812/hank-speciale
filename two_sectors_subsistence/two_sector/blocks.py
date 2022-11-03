import numpy as np
import numba as nb

from GEModelTools import lag, lead
   
@nb.njit
def block_pre(par,ini,ss,path,ncols=1):

    for ncol in nb.prange(ncols):

        adjcost_N = path.adjcost_N[ncol,:]
        adjcost_L = path.adjcost_L[ncol,:]
        adjcost = path.adjcost[ncol,:]
        A_hh = path.A_hh[ncol,:]
        A = path.A[ncol,:]
        B = path.B[ncol,:]
        C = path.C[ncol,:]
        C_hh = path.C_hh[ncol,:]
        clearing_C = path.clearing_C[ncol,:]
        C_N = path.C_N[ncol,:]
        C_N_hh = path.C_N_hh[ncol,:]
        clearing_C_N = path.clearing_C_N[ncol,:]
        C_L = path.C_L[ncol,:]                
        C_L_hh = path.C_L_hh[ncol,:]
        clearing_C_L = path.clearing_C_L[ncol,:]
        clearing_A = path.clearing_A[ncol,:]
        clearing_N = path.clearing_N[ncol,:]
        d = path.d[ncol,:]
        d_N = path.d_N[ncol,:]
        d_L = path.d_L[ncol,:]
        G = path.G[ncol,:]
        i = path.i[ncol,:]
        N_hh = path.N_hh[ncol,:]
        N = path.N[ncol,:]
        N_N = path.N[ncol,:]                
        N_L = path.N[ncol,:]
        #M = path.M[ncol,:]
        M_N = path.M_N[ncol,:]                
        M_L = path.M_L[ncol,:]
        pm_L = path.pm_L[ncol,:]        
        pm_N = path.pm_N[ncol,:]        
        NKPC_res_N = path.NKPC_res_N[ncol,:]
        NKPC_res_L = path.NKPC_res_L[ncol,:]
        #NKPC_res = path.NKPC_res[ncol,:]
        pi = path.pi[ncol,:]
        pi_N = path.pi_N[ncol,:]
        pi_L = path.pi_L[ncol,:]
        r = path.r[ncol,:]
        rstar = path.rstar[ncol,:]
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
        Z_N = path.Z_N[ncol,:]
        Z_L = path.Z_L[ncol,:]
        Q = path.Q[ncol,:]
        P = path.P[ncol,:]

        #################
        # implied paths #
        #################

        # a. firms
        
        
        P[:] = (par.alpha_hh+Q**(1-par.gamma_hh)*(1-par.alpha_hh))**(1/(1-par.gamma_hh)) 

        
        
        
        #w_L[:] = (Z_L)*((par.mu_L**(par.gamma_L-1)-par.alpha_L*pm**(1-par.gamma_L))/(1-par.alpha_L))**(1/(1-par.gamma_L))
        #w_N[:] = Q*w_L
        #pm_N[:] = Q*pm
        #Z_N[:] = w_N*((par.mu_N**(par.gamma_N-1)-par.alpha_N*pm_N**(1-par.gamma_N))/(1-par.alpha_N))**(-1/(1-par.gamma_N))
        #Y_N[:] = Y*P-Q*Y_L
        #mc_N[:] = ((1-par.alpha_N)*(w_N/Z_N)**(1-par.gamma_N)+par.alpha_N*pm_N**(1-par.gamma_N))**(1/(1-par.gamma_N))
        #mc_L[:] = ((1-par.alpha_L)*(w_L/Z_L)**(1-par.gamma_L)+par.alpha_L*pm**(1-par.gamma_L))**(1/(1-par.gamma_L))
        #
        #M_L[:] = (par.alpha_L*(pm/mc_L)**(-par.gamma_L)*Y_L)    
        #N_L[:] = (1-par.alpha_L)*(w_L/mc_L)**(-par.gamma_L)*Z_L**(par.gamma_L-1)*Y_L    
        #
        #M_N[:] = par.alpha_N*(pm_N/mc_N)**(-par.gamma_N)*Y_N    
        #N_N[:] = (1-par.alpha_N)*(w_N/mc_N)**(-par.gamma_N)*Z_N**(par.gamma_N-1)*Y_N

        '''
        # sector N
        mc_N[:] = ((1-par.alpha_N)*(w_N/Z_N)**(1-par.gamma_N)+par.alpha_N*pm**(1-par.gamma_N))**(1/(1-par.gamma_N))
        N_N[:] = (w_N/mc_N)**(-par.gamma_N)*(1-par.alpha_N)*Z_N**(par.gamma_N-1)*Y_N
        M_N[:] = par.alpha_N*(pm/mc_N)**(-par.gamma_N)*Y_N 
        adjcost_N[:] = par.mu_N/(par.mu_N-1)/(2*par.kappa_N)*np.log(1+pi_N)**2*Y_N
        d_N[:] = Y_N-w_N*N_N-pm*M_N-adjcost_N


        # sector L
        mc_L[:] = ((1-par.alpha_L)*(w_L/Z_L)**(1-par.gamma_L)+par.alpha_L*pm**(1-par.gamma_L))**(1/(1-par.gamma_L))
        N_L[:] = (1-par.alpha_L)*(w_L/mc_L)**(-par.gamma_L)*Z_L**(par.gamma_L-1)*Y_L 
        M_L[:] = (par.alpha_L*(pm/mc_L)**(-par.gamma_L)*Y_L) 
        adjcost_L[:] = par.mu_L/(par.mu_L-1)/(2*par.kappa_L)*np.log(1+pi_L)**2*Y_L
        d_L[:] = Y_L-w_L*N_L-pm*M_L-adjcost_L
        '''
        # inflation
        Q_lag = lag(ini.Q,Q)
        pi_L[:] = (Q/Q_lag)*(1+pi_N)-1
        pi[:] = pi_N**par.epsilon*pi_L**(1-par.epsilon) #preliminary inflation indexing
        
        # prices
        w_L[:] = (1/Q)*w_N
        pm_L[:] = (1/Q)*pm_N

        # production
        mc_N[:] = ((1-par.alpha_N)*(w_N/Z_N)**(1-par.gamma_N)+par.alpha_N*pm_N**(1-par.gamma_N))**(1/(1-par.gamma_N))
        mc_L[:] = ((1-par.alpha_L)*(w_L/Z_L)**(1-par.gamma_L)+par.alpha_L*pm_L**(1-par.gamma_L))**(1/(1-par.gamma_L))

        adjcost_N[:] = (par.mu_N/(par.mu_N-1))*(1/(2*par.kappa_N))*(np.log(1+pi_N))**2
        adjcost_L[:] = (par.mu_L/(par.mu_L-1))*(1/(2*par.kappa_L))*(np.log(1+pi_L))**2

        M_N[:] = (1-par.alpha_N)**(1/(par.gamma_N))*(Z_N*N_N)*(par.alpha_N**((1-par.gamma_N)/par.gamma_N)*(pm_N/mc_N)**(par.gamma_N-1)-par.alpha_N**(1/par.gamma_N))**(par.gamma_N/(1-par.gamma_N))
        M_L[:] = (1-par.alpha_L)**(1/(par.gamma_L))*(Z_L*N_L)*(par.alpha_L**((1-par.gamma_L)/par.gamma_L)*(pm_L/mc_L)**(par.gamma_L-1)-par.alpha_L**(1/par.gamma_L))**(par.gamma_L/(1-par.gamma_L))

        Y_N[:] = (par.alpha_N**(1/par.gamma_N)*M_N**((par.gamma_N-1)/par.gamma_N)+(1-par.alpha_N)*(1/par.gamma_N)*(Z_N*N_N)**((par.gamma_N-1)/(par.gamma_N)))**(par.gamma_N/(par.gamma_N-1))
        Y_L[:] = (par.alpha_L**(1/par.gamma_L)*M_L**((par.gamma_L-1)/par.gamma_L)+(1-par.alpha_L)*(1/par.gamma_L)*(Z_L*N_L)**((par.gamma_L-1)/(par.gamma_L)))**(par.gamma_L/(par.gamma_L-1))
        
        # b. monetary policy
        rstar[:] = ss.r
        istar[:] = pi + rstar
        i[:] = istar + par.phi*pi + par.phi_y*(Y-(ss.Y_N+Q*ss.Y_L))
        i_lag = lag(ini.i,i)
        r[:] = (1+i_lag)/(1+pi)-1 ## Fix these taylor rule weights 

        # c. government
        B[:] = ss.B
        tau[:] = r*B
        G[:] = tau-r*B
        
        # d. aggregates
        A[:] = B[:] = ss.B
        C_N[:] = Y_N-adjcost_N-pm_N*M_N
        #C_L[:] = Y_L-adjcost_L-pm*M_L
        C[:] = (C_N + Q*C_L)/P
        N[:] = N_N + N_L

@nb.njit
def block_post(par,ini,ss,path,ncols=1):

    for ncol in nb.prange(ncols):

        adjcost_N = path.adjcost_N[ncol,:]
        adjcost_L = path.adjcost_L[ncol,:]
        adjcost = path.adjcost[ncol,:]
        A_hh = path.A_hh[ncol,:]
        A = path.A[ncol,:]
        B = path.B[ncol,:]
        C = path.C[ncol,:]
        C_hh = path.C_hh[ncol,:]
        clearing_C = path.clearing_C[ncol,:]
        C_N = path.C_N[ncol,:]
        C_N_hh = path.C_N_hh[ncol,:]
        clearing_C_N = path.clearing_C_N[ncol,:]
        C_L = path.C_L[ncol,:]       
        C_L_hh = path.C_L_hh[ncol,:]
        clearing_C_L = path.clearing_C_L[ncol,:]
        clearing_A = path.clearing_A[ncol,:]
        clearing_N = path.clearing_N[ncol,:]
        d = path.d[ncol,:]
        d_N = path.d_N[ncol,:]
        d_L = path.d_L[ncol,:]
        G = path.G[ncol,:]
        i = path.i[ncol,:]
        N_hh = path.N_hh[ncol,:]
        N = path.N[ncol,:]
        M_N = path.M_N[ncol,:]                
        M_L = path.M_L[ncol,:]
        #M = path.M[ncol,:]
        pm = path.pm[ncol,:]        
        pm_N = path.pm_N[ncol,:]        
        NKPC_res_N = path.NKPC_res_N[ncol,:]
        NKPC_res_L = path.NKPC_res_L[ncol,:]
        #NKPC_res = path.NKPC_res[ncol,:]
        pi = path.pi[ncol,:]
        pi_N = path.pi_N[ncol,:]
        pi_L = path.pi_L[ncol,:]
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
        Z_N = path.Z_N[ncol,:]
        Z_L = path.Z_L[ncol,:]
        Q = path.Q[ncol,:]
        P = path.P[ncol,:]

        #################
        # check targets #
        #################

        # a. phillips curve
        r_plus = lead(r,ss.r)
        pi_N_plus = lead(pi_N,ss.pi_N)
        pi_L_plus = lead(pi_L,ss.pi_L)
        Y_N_plus = lead(Y_N,ss.Y_N)
        Y_L_plus = lead(Y_L,ss.Y_L)
        Q_lag = lag(ini.Q,Q)
        pi_L[:] = Q/Q_lag*pi_N


        mc_N[:] = ((1-par.alpha_N)*(w_N/Z_N)**(1-par.gamma_N)+par.alpha_N*pm_N**(1-par.gamma_N))**(1/(1-par.gamma_N)) 
        mc_L[:] = ((1-par.alpha_L)*(w_L*Z_L)**(1-par.gamma_L)+par.alpha_L*pm**(1-par.gamma_L))**(1/(1-par.gamma_L)) 
        
        NKPC_res_N[:] = par.kappa_N*(mc_N-1/par.mu_N) + Y_N_plus/Y_N*np.log(1+pi_N_plus)/(1+r_plus) - np.log(1+pi_N)
        NKPC_res_L[:] = par.kappa_L*(mc_L-1/par.mu_L) + Y_L_plus/Y_L*np.log(1+pi_L_plus)/(1+r_plus) - np.log(1+pi_L)

        # b. market clearing
        clearing_A[:] = A-A_hh
        clearing_C_N[:] = C_N-C_N_hh
        clearing_C_L[:] = C_L-C_L_hh
        clearing_N[:] = N-N_hh
        clearing_C[:] = C-C_hh