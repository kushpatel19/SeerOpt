import os
import numpy as np
from trajectory_utils.plot_trajectory import plot_trajectory,plot_actual_trajectory
from trajectory_utils.mpc import nmpc_controller
from trajectory_utils.sim import *
import random
import shutil

def main():
    # Path to trajectory file
    trajectories_path = os.path.join(os.getcwd(),'exp','eval_navhard','2025.11.27.21.00.17')
    trajectory_file_name = "2025.11.27.21.03.43_trajectories.npy"
    trajectory_file_path = os.path.join(trajectories_path, trajectory_file_name)

    # Load trajectories dict
    trajectories = np.load(trajectory_file_path, allow_pickle=True).item()

    # Pick 5 random keys
    # # random_keys = random.sample(list(trajectories.keys()), min(5, len(trajectories)))
    # random_keys = ["60216ba3ee9557d9"]  # for consistent testing


    # shutil.rmtree("./trajectory_videos")
    # shutil.rmtree("./trajectory_plots")

    # Dictionary to store all MPC trajectories
    mpc_trajectories = {}

    for traj_key in trajectories.keys():
        print(f"Processing trajectory key: {traj_key}")
        trajectory = trajectories[traj_key]  # shape (T, 3)
        prob, N_mpc, n_x, n_g, n_p, lb_var, ub_var, lb_cons, ub_cons = nmpc_controller(trajectory)
        # random initialization
        x0_nlp    = np.random.randn(n_x, 1) * 0.025
        lamx0_nlp = np.random.randn(n_x, 1) * 0.025
        lamg0_nlp = np.random.randn(n_g, 1) * 0.025
        init =  [0.1, 0.0, 0.0, 0.0, 0.0,  0.0]
        init[3] = trajectory[0, 0]     # x position
        init[4] = trajectory[0, 1]     # y position
        init[5] = trajectory[0, 2]     # heading phi
        opts = {'ipopt.max_iter':3000}
        solver = ca.nlpsol('solver', 'ipopt', prob , opts)
        sol = solver(x0=x0_nlp, lam_x0=lamx0_nlp, lam_g0=lamg0_nlp,
                            lbx=lb_var, ubx=ub_var, lbg=lb_cons, ubg=ub_cons, p=init)
        x0_nlp    = sol["x"].full()
        lamx0_nlp = sol["lam_x"].full()
        lamg0_nlp = sol["lam_g"].full() # np.zeros((n_g, 1))
    
        T_sim  = 4 
        dt_ctrl = 0.1
        rk_interval = 10
        N_sim = int(np.ceil(T_sim / dt_ctrl))
        integrator, car_dynamics = GetCarModel(None, ts = np.linspace(0, dt_ctrl, rk_interval), ode = "rk")
        sim_time = np.linspace(0, T_sim, N_sim * (rk_interval - 1) + 1)
        y0 = np.reshape(init, (6, 1))
        is_silent = False# change this to False if you want to see all errors and warnings
        x_log, u_log, tire_force_log = SimVehicle(y0, nmpc_controller, integrator, car_dynamics, N_sim , True, x0_nlp, lamx0_nlp, lamg0_nlp, is_silent,trajectory)
        
        # Extract x, y, heading and store in dictionary
        mpc_trajectory = x_log[3:6, :].T  # shape (N, 3) - x, y, heading
        mpc_trajectories[traj_key] = mpc_trajectory
        
        plot_actual_trajectory(trajectory, traj_key)
        plot_trajectory(trajectory, traj_key, mpc_trajectory)
    
    # Save all MPC trajectories to a single .npy file
    mpc_traj_save_path = os.path.join(os.getcwd(), "mpc_trajectory.npy")
    np.save(mpc_traj_save_path, mpc_trajectories)
    print(f"\nAll MPC trajectories saved to: {mpc_traj_save_path}")
    print(f"Total trajectories saved: {len(mpc_trajectories)}")
    
if __name__ == "__main__":
    main()