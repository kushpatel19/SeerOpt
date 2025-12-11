import matplotlib.pyplot as plt
import os 
import numpy as np
def plot_actual_trajectory(trajectory,trajectory_key):
    fig_save_path = os.path.join(os.getcwd(), "trajectory_plots", trajectory_key)
    os.makedirs(fig_save_path, exist_ok=True)
    T_ref = trajectory.shape[0]
    # Generate time axes (assuming equal spacing)
    time_ref = np.arange(T_ref)

    # Extract reference signals
    x_ref = trajectory[:, 0]
    y_ref = trajectory[:, 1]
    h_ref = trajectory[:, 2]
    plt.figure()

    # Plot reference x, MPC x
    plt.subplot(4, 1, 1)
    plt.plot(time_ref, x_ref, label="Ref X")
    plt.xlabel("Index (time step)")
    plt.ylabel("X")
    plt.legend()

    # Plot reference y, MPC y
    plt.subplot(4, 1, 2)
    plt.plot(time_ref, y_ref, label="Ref Y")
    plt.xlabel("Index (time step)")
    plt.ylabel("Y")
    plt.legend()

    # Plot reference heading, MPC heading
    plt.subplot(4, 1, 3)
    plt.plot(time_ref, h_ref, label="Ref θ")
    plt.xlabel("Index")
    plt.ylabel("Heading")
    plt.legend()

    # Plot X–Y plane for reference and MPC
    plt.subplot(4, 1, 4)
    plt.plot(x_ref, y_ref, label="Reference path")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_path, f"actual_trajectory_plot.png"))
    plt.close()


def plot_trajectory(trajectory, trajectory_key,mpc_trajectory=None):
    """Takes in a trajectory array and plots x vs time, y vs time, heading vs time, and x vs y with MPC overlay."""

    fig_save_path = os.path.join(os.getcwd(), "trajectory_plots", trajectory_key)
    os.makedirs(fig_save_path, exist_ok=True)

    T_ref = trajectory.shape[0]
    if mpc_trajectory is not None:
        T_mpc = mpc_trajectory.shape[0]
        time_mpc = np.linspace(0, 4, T_mpc)
        # Extract MPC rollout signals
        x_mpc = mpc_trajectory[:, 0]
        y_mpc = mpc_trajectory[:, 1]
        h_mpc = mpc_trajectory[:, 2]


    # Generate time axes (assuming equal spacing)
    time_ref  = np.linspace(0, 4, T_ref)

    # Extract reference signals
    x_ref = trajectory[:, 0]
    y_ref = trajectory[:, 1]
    h_ref = trajectory[:, 2]

    # 
    plt.figure()

    # Plot reference x, MPC x
    plt.subplot(4, 1, 1)
    plt.plot(time_ref, x_ref, label="Ref X")
    if mpc_trajectory is not None:
        plt.plot(time_mpc, x_mpc, label="MPC X")
    plt.xlabel("Index (time step)")
    plt.ylabel("X")
    plt.legend()

    # Plot reference y, MPC y
    plt.subplot(4, 1, 2)
    plt.plot(time_ref, y_ref, label="Ref Y")
    if mpc_trajectory is not None:
        plt.plot(time_mpc, y_mpc, label="MPC Y")
    plt.xlabel("Index (time step)")
    plt.ylabel("Y")
    plt.legend()

    # Plot reference heading, MPC heading
    plt.subplot(4, 1, 3)
    plt.plot(time_ref, h_ref, label="Ref θ")
    if mpc_trajectory is not None:
        plt.plot(time_mpc, h_mpc, label="MPC θ")
    plt.xlabel("Index")
    plt.ylabel("Heading")
    plt.legend()

    # Plot X–Y plane for reference and MPC
    plt.subplot(4, 1, 4)
    plt.plot(x_ref, y_ref, label="Reference path")
    if mpc_trajectory is not None:
        plt.plot(x_mpc, y_mpc, label="MPC rollout")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_path, f"trajectory_plot.png"))
    plt.close()
