from trajectory_utils.sim import *
from trajectory_utils.utils import *

def nmpc_controller(ref_trajectory):
    target = ref_trajectory[-1,:]
    T = 4 # prediction horizon
    dt = 0.1 # time step
    N = int(T / dt)  # number of steps in the horizon

    Dim_state = 6 # (U_xk, U_yk, r, x_k, y_k, phi_k)
    Dim_ctrl  = 2 # (F_xk, gamma_k)
    Dim_aux   = 4 # (F_yf, F_yr, F_muf (slack front), F_mur (slack rear))

    # casadi variable for state, control, and auxiliary
    xm = ca.MX.sym('xm', (Dim_state, 1)) 
    um = ca.MX.sym('um', (Dim_ctrl, 1))
    aux = ca.MX.sym('aux', (Dim_aux, 1))

    # renaming the control inputs for better readability
    Fx,delta = um[0], um[1]

    # Air drag and rolling resistance
    Fd = param["Frr"] + param["Cd"] * xm[0]**2
    Fd = Fd * ca.tanh(- xm[0] * 100)
    Fb = 0.0

    # front and rear slip angles
    af ,ar = get_slip_angle(xm[0] , xm[1] , delta , delta , param)

    # front and rear normal loads
    Fzf , Fzr = normal_load(Fx, param)

    # front and rear longitudinal forces
    Fxf, Fxr = chi_fr(Fx)

    # front and rear lateral forces
    Fyf = tire_model_ctrl(af, Fzf, Fxf, param["C_alpha_f"], param["mu_f"] )
    Fyr = tire_model_ctrl(ar, Fzr, Fxr, param["C_alpha_r"], param["mu_r"] )

    # state derivatives
    # x-acceleration
    dUx = (Fxf * ca.cos(delta) - aux[0] * ca.sin(delta) + Fxr + Fd) / param["m"] + xm[2] * xm[1]
    # y-acceleration
    dUy = (aux[0] * ca.cos(delta) + Fxf * ca.sin(delta) + aux[1] + Fb) / param["m"] - xm[2] * xm[0] 
    # yaw acceleration                 
    dr = (param["L_f"] * (aux[0] * ca.cos(delta) + Fxf * ca.sin(delta)) - param["L_r"] * aux[1]) / param["Izz"] 

    # x velocity
    dx = ca.cos(xm[5]) * xm[0] - ca.sin(xm[5]) * xm[1]
    # y velocity
    dy = ca.sin(xm[5]) * xm[0] + ca.cos(xm[5]) * xm[1]
    # yaw rate
    dyaw = xm[2]

    xdot = ca.vertcat(dUx, dUy, dr, dx, dy, dyaw)

    xkp1 = xm + xdot * dt
    Fun_dynamics_dt = ca.Function('f_dt', [xm, um, aux], [xkp1])

    # enforce contraints for auxiliary variables
    alg = ca.vertcat(aux[0] - Fyf, aux[1] - Fyr)
    Fun_alg = ca.Function('alg', [xm, um, aux], [alg])

    # """" MPC Variables """"""""
    # state, control, auxiliary over the prediction horizon
    x = ca.MX.sym('x', (Dim_state, N + 1))  # (U_xk, U_yk, r, x_k, y_k, phi_k)
    u = ca.MX.sym('u', (Dim_ctrl, N))       # (F_xk, gamma_k)
    z = ca.MX.sym('z', (Dim_aux, N))        # (F_yf, F_yr, F_muf (slack front), F_mur (slack rear))
    # initial state parameter
    p = ca.MX.sym('p', (Dim_state, 1))

    ###################### MPC constraints start ######################
    ## MPC equality constraints ##
    cons_dynamics = []
    for k in range(N):
        xkp1 = Fun_dynamics_dt(x[:, k], u[:, k], z[:, k])
        # Fy2 = Fun_alg(x[:, k], u[:, k], z[:, k])
        Fy2  = Fun_alg(x[:, k], u[:, k], z[:, k])
        for j in range(Dim_state):
            cons_dynamics.append(x[j, k+1] - xkp1[j])
        for j in range(2):
            cons_dynamics.append(Fy2[j])

    ## MPC inequality constraints ##
    cons_ineq = []
    for k in range(N):
        cons_ineq.append(2 - x[0, k])  # U_xk <= 2 m/s
        cons_ineq.append(u[0, k] * x[0, k] - param["Peng"]) # Fxk * U_xk <= Peng engine power limits
    
    for k in range(N):
        Fx,delta = u[0,k], u[1,k]
        af,ar = get_slip_angle(x[0,k], x[1,k], x[2,k], delta, param)
        Fxf, Fxr = chi_fr(Fx)
        Fzf, Fzr = normal_load(Fx, param)

        Fyf = tire_model_ctrl(af, Fzf, Fxf, param["C_alpha_f"], param["mu_f"] )
        Fyr = tire_model_ctrl(ar, Fzr, Fxr, param["C_alpha_r"], param["mu_r"] )

        # front tire limits
        cons_ineq.append((Fyf**2 + Fxf**2) - (param["mu_f"]*Fzf)**2 - z[2, k]**2)
        # rear tire limits
        cons_ineq.append((Fyr**2 + Fxr**2) - (param["mu_r"]*Fzr)**2 - z[3, k]**2)

    ###################### MPC cost start ######################
    cost = 0

    
    # cost += 200* x[5, N]**2 # phi -> 0
    # U_x should be high
    # cost -= 200 * (x[0, N])**2  # U_x -> inf
    # cost += 20.0 * x[5, k]**2  # phi_k -> 0
    # x should reach target x position

    cost += 350.0 * (target[1] - x[4, N])**2  # final y position error
    cost += 450.0 * (target[0] - x[3, N])**2 # final x position error

    ## Stage costs (at k)
    for k in range(N):
        cost += 70 * (target[1] - x[4, k])**2 # y position error
        cost += 70 * (target[0] - x[3, k])**2 # x position error

        # cost += 20.0 * x[5, k]**2  # phi_k -> 0
        # cost -= 20.0 * (x[0, k])**2    # Ux_k -> inf
        cost += 5* u[0, k]**2  # F_xk
        cost += 5 * u[1, k]**2   # gamma_k
        
    ## Excessive slip angle / friction
    for k in range(N):
        Fx = u[0, k]; delta = u[1, k]
        af, ar = get_slip_angle(x[0, k], x[1, k], x[2, k], delta, param)
        Fzf, Fzr= normal_load(Fx, param)
        Fxf, Fxr = chi_fr(Fx)

        xi = 0.85
        F_offset = 2000   ## A slacked ReLU function using only sqrt()
        Fyf_max_sq = (param["mu_f"] * Fzf)**2 - (0.999 * Fxf)**2
        Fyf_max_sq = (ca.sqrt( Fyf_max_sq**2 + F_offset) + Fyf_max_sq) / 2
        Fyf_max = ca.sqrt(Fyf_max_sq)

        ## Modified front slide sliping angle
        alpha_mod_f = ca.arctan(3 * Fyf_max / param["C_alpha_f"] * xi)

        Fyr_max_sq = (param["mu_f"] * Fzf)**2 - (0.999 * Fxf)**2
        Fyr_max_sq = (ca.sqrt( Fyr_max_sq**2 + F_offset) + Fyr_max_sq) / 2
        Fyr_max = ca.sqrt(Fyr_max_sq)

        ## Modified rear slide sliping angle
        alpha_mod_r = ca.arctan(3 * Fyr_max / param["C_alpha_r"] * xi)

        ## Limit friction penalty

        W_alpha = 100.0
        cost += W_alpha * ca.if_else(ca.fabs(af) >= alpha_mod_f, (ca.fabs(af) - alpha_mod_f)**2, 0.0)  
        cost += W_alpha * ca.if_else(ca.fabs(ar) >= alpha_mod_r, (ca.fabs(ar) - alpha_mod_r)**2, 0.0)  
        cost += 1000.0 * (z[2, k]**2 + z[3, k]**2)

    # Initial condition as parameters
    cons_init = [x[:, 0] - p]
    ub_init_cons = np.zeros((Dim_state, 1))
    lb_init_cons = np.zeros((Dim_state, 1))

    state_ub = np.array([ 1e2,  1e2,  1e2,  1e8,  1e8,  1e8])
    state_lb = np.array([-1e2, -1e2, -1e2, -1e8, -1e8, -1e8])
    
    ## Set the control limits for upper and lower bounds
    ctrl_ub  = np.array([1e5, param["delta_max"]])          # (traction force, steering angle)
    ctrl_lb  = np.array([-1e5, -param["delta_max"]])        # (-traction force, -steering angle)
    
    aux_ub   = np.array([ 1e5,  1e5,  1e5,  1e5])
    aux_lb   = np.array([-1e5, -1e5, -1e5, -1e5])

    lb_dynamics = np.zeros((len(cons_dynamics), 1))
    ub_dynamics = np.zeros((len(cons_dynamics), 1))

    lb_ineq = np.zeros((len(cons_ineq), 1)) - 1e9
    ub_ineq = np.zeros((len(cons_ineq), 1))

    ub_x = np.matlib.repmat(state_ub, N + 1, 1)
    lb_x = np.matlib.repmat(state_lb, N + 1, 1)
    ub_u = np.matlib.repmat(ctrl_ub, N, 1)
    lb_u = np.matlib.repmat(ctrl_lb, N, 1)
    ub_z = np.matlib.repmat(aux_ub, N, 1)
    lb_z = np.matlib.repmat(aux_lb, N, 1)

    lb_var = np.concatenate((lb_u.reshape((Dim_ctrl * N, 1)), 
                             lb_x.reshape((Dim_state * (N+1), 1)),
                             lb_z.reshape((Dim_aux * N, 1))
                             ))

    ub_var = np.concatenate((ub_u.reshape((Dim_ctrl * N, 1)), 
                             ub_x.reshape((Dim_state * (N+1), 1)),
                             ub_z.reshape((Dim_aux * N, 1))
                             ))

    vars_NLP   = ca.vertcat(u.reshape((Dim_ctrl * N, 1)), x.reshape((Dim_state * (N+1), 1)), z.reshape((Dim_aux * N, 1)))
    cons_NLP = cons_dynamics + cons_ineq + cons_init
    cons_NLP = ca.vertcat(*cons_NLP)
    lb_cons = np.concatenate((lb_dynamics, lb_ineq, lb_init_cons))
    ub_cons = np.concatenate((ub_dynamics, ub_ineq, ub_init_cons))

    prob = {"x": vars_NLP, "p":p, "f": cost, "g":cons_NLP}

    return prob, N, vars_NLP.shape[0], cons_NLP.shape[0], p.shape[0], lb_var, ub_var, lb_cons, ub_cons                                           
