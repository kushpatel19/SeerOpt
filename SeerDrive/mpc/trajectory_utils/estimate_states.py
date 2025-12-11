"""
Helper functions to estimate vehicle states from trajectory poses (x, y, heading).

Since trajectory data only contains [x, y, heading], we need to estimate:
- Velocity (v_x)
- Acceleration (a_x)
- Steering angle (δ)
- Steering rate (δ_dot)
"""

import numpy as np
from typing import Tuple


def estimate_velocity_from_poses(poses: np.ndarray, dt: float = 0.5) -> np.ndarray:
    """
    Estimate velocity from position measurements.
    
    Args:
        poses: (N, 3) array of [x, y, heading]
        dt: Time step [s]
    
    Returns:
        v_x: (N-1,) array of velocity estimates [m/s]
    """
    dx = np.diff(poses[:, 0])
    dy = np.diff(poses[:, 1])
    displacement = np.sqrt(dx**2 + dy**2)
    v_x = displacement / dt
    return v_x


def estimate_acceleration_from_velocity(v_x: np.ndarray, dt: float = 0.5) -> np.ndarray:
    """
    Estimate acceleration from velocity.
    
    Args:
        v_x: (N,) array of velocities [m/s]
        dt: Time step [s]
    
    Returns:
        a_x: (N-1,) array of acceleration estimates [m/s²]
    """
    a_x = np.diff(v_x) / dt
    return a_x


def estimate_steering_angle_from_heading(
    poses: np.ndarray, v_x: np.ndarray, dt: float = 0.5, L: float = 3.0
) -> np.ndarray:
    """
    Estimate steering angle from heading rate using bicycle model.
    
    From: θ_dot = (v_x * tan(δ)) / L
    Solving for δ: δ = atan(θ_dot * L / v_x)
    
    Args:
        poses: (N, 3) array of [x, y, heading]
        v_x: (N-1,) array of velocities [m/s]
        dt: Time step [s]
        L: Wheelbase length [m]
    
    Returns:
        delta: (N-1,) array of steering angle estimates [rad]
    """
    # Compute heading rate
    dtheta = np.diff(poses[:, 2])
    # Normalize angle differences to [-π, π]
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
    theta_dot = dtheta / dt
    
    # Avoid division by zero for low speeds
    v_x_safe = np.maximum(v_x, 0.1)  # Minimum 0.1 m/s
    delta = np.arctan(theta_dot * L / v_x_safe)
    
    return delta


def estimate_steering_rate_from_steering_angle(
    delta: np.ndarray, dt: float = 0.5
) -> np.ndarray:
    """
    Estimate steering rate from steering angle.
    
    Args:
        delta: (N,) array of steering angles [rad]
        dt: Time step [s]
    
    Returns:
        delta_dot: (N-1,) array of steering rate estimates [rad/s]
    """
    delta_dot = np.diff(delta) / dt
    return delta_dot


def estimate_states_from_poses(
    poses: np.ndarray, dt: float = 0.5, L: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate full state vector from pose trajectory.
    
    Args:
        poses: (N, 3) array of [x, y, heading]
        dt: Time step [s]
        L: Wheelbase length [m]
    
    Returns:
        states: (N-1, 5) array of [x, y, θ, v_x, δ]
        controls: (N-2, 2) array of [a_x, δ_dot] (estimated control inputs)
    """
    N = len(poses)
    
    # Estimate velocity
    v_x = estimate_velocity_from_poses(poses, dt)
    
    # Estimate acceleration
    a_x = estimate_acceleration_from_velocity(v_x, dt)
    
    # Estimate steering angle
    delta = estimate_steering_angle_from_heading(poses, v_x, dt, L)
    
    # Estimate steering rate
    delta_dot = estimate_steering_rate_from_steering_angle(delta, dt)
    
    # Construct state array: [x, y, θ, v_x, δ]
    states = np.zeros((N-1, 5))
    states[:, 0] = poses[:-1, 0]  # x
    states[:, 1] = poses[:-1, 1]   # y
    states[:, 2] = poses[:-1, 2]   # θ
    states[:, 3] = v_x             # v_x
    states[:, 4] = delta           # δ
    
    # Construct control array: [a_x, δ_dot]
    controls = np.zeros((N-2, 2))
    controls[:, 0] = a_x           # a_x
    controls[:, 1] = delta_dot     # δ_dot
    
    return states, controls


def estimate_states_simple(poses: np.ndarray, dt: float = 0.5) -> np.ndarray:
    """
    Simple state estimation returning only [x, y, θ, v_x, δ].
    Useful for MPC with reduced state model.
    
    Args:
        poses: (N, 3) array of [x, y, heading]
        dt: Time step [s]
    
    Returns:
        states: (N-1, 5) array of [x, y, θ, v_x, δ]
    """
    N = len(poses)
    states = np.zeros((N-1, 5))
    
    # Position and heading (directly from poses)
    states[:, 0] = poses[:-1, 0]  # x
    states[:, 1] = poses[:-1, 1]   # y
    states[:, 2] = poses[:-1, 2]   # θ
    
    # Estimate velocity from displacement
    dx = np.diff(poses[:, 0])
    dy = np.diff(poses[:, 1])
    displacement = np.sqrt(dx**2 + dy**2)
    v_x = displacement / dt
    states[:, 3] = v_x
    
    # Estimate steering angle from heading rate
    dtheta = np.diff(poses[:, 2])
    # Normalize angle differences to [-π, π]
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
    theta_dot = dtheta / dt
    
    # Avoid division by zero for low speeds
    L = 3.0  # Default wheelbase
    v_x_safe = np.maximum(v_x, 0.1)  # Minimum 0.1 m/s
    delta = np.arctan(theta_dot * L / v_x_safe)
    states[:, 4] = delta
    
    return states


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Example trajectory (8 poses, 0.5s intervals)
    trajectory = np.array([
        [1.982, 0.092, 0.081],
        [3.917, 0.286, 0.129],
        [5.718, 0.554, 0.170],
        [7.316, 0.849, 0.205],
        [8.669, 1.115, 0.215],
        [9.731, 1.334, 0.207],
        [10.528, 1.505, 0.227],
        [11.130, 1.634, 0.245],
    ])
    
    print("Input trajectory shape:", trajectory.shape)
    print("\nEstimating states...")
    
    states, controls = estimate_states_from_poses(trajectory, dt=0.5, L=3.0)
    
    print(f"\nEstimated states shape: {states.shape}")
    print("States (x, y, θ, v_x, δ):")
    print(states)
    
    print(f"\nEstimated controls shape: {controls.shape}")
    print("Controls (a_x, δ_dot):")
    print(controls)
    
    print("\nState statistics:")
    print(f"  Velocity range: [{states[:, 3].min():.3f}, {states[:, 3].max():.3f}] m/s")
    print(f"  Steering angle range: [{states[:, 4].min():.3f}, {states[:, 4].max():.3f}] rad")
    print(f"  Acceleration range: [{controls[:, 0].min():.3f}, {controls[:, 0].max():.3f}] m/s²")


