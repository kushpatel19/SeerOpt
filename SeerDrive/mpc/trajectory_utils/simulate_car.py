import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from matplotlib import animation
from PIL import Image

def wrap_interp(theta1, theta2, t):
    diff = (theta2 - theta1 + np.pi) % (2 * np.pi) - np.pi
    return theta1 + t * diff

def interpolate_trajectory(traj, num_points=150):
    traj = np.array(traj)
    thetas = traj[:, 2]

    t_ref = np.arange(len(traj))
    t_interp = np.linspace(0, len(traj) - 1, num_points)

    interp_x = np.interp(t_interp, t_ref, traj[:, 0])
    interp_y = np.interp(t_interp, t_ref, traj[:, 1])

    interp_theta = []
    for t in t_interp:
        i = int(np.floor(t))
        alpha = t - i
        th = wrap_interp(thetas[i], thetas[min(i+1, len(traj)-1)], alpha)
        interp_theta.append(th)

    return interp_x, interp_y, np.array(interp_theta)

def simulate_car(traj_8, save_folder="simulation_output", fps=20, car_img_path=os.path.join(os.getcwd(),"red car.png")):
    # 1. Interpolate trajectory
    interp_x, interp_y, interp_theta = interpolate_trajectory(traj_8)

    # 2. Create folder for frames
    frames_folder = os.path.join(save_folder, "frames")
    os.makedirs(frames_folder, exist_ok=True)

    # 3. Load car image
    car_path = os.path.join(save_folder, car_img_path)
    car_img = Image.open(car_path)
    car_arr = np.array(car_img)

    # 4. Plot background path once
    bg_x, bg_y = interp_x, interp_y

    frame_paths = []
    for i in range(len(interp_x)):
        fig, ax = plt.subplots(figsize=(5,5))
        ax.set_xlim(np.min(bg_x)-1, np.max(bg_x)+1)
        ax.set_ylim(np.min(bg_y)-1, np.max(bg_y)+1)
        ax.set_aspect("equal")
        ax.grid(True)
        ax.plot(bg_x, bg_y, "k--", alpha=0.3)

        # Car pose
        x, y, theta = interp_x[i], interp_y[i], interp_theta[i]

        # Rotate car image
        car_rot = Image.fromarray(car_arr).rotate(-np.degrees(theta), expand=True)
        car_rot_arr = np.array(car_rot)

        # Overlay car
        imgbox = ax.imshow(car_rot_arr, extent=[x-0.5, x+0.5, y-0.5, y+0.5], zorder=5)

        # Save frame
        frame_path = os.path.join(frames_folder, f"frame_{i:04d}.png")
        fig.savefig(frame_path, dpi=120)
        plt.close(fig)
        frame_paths.append(frame_path)

    # 5. Stitch frames to video using OpenCV
    frame_files = sorted(os.listdir(frames_folder))
    frame_paths = [os.path.join(frames_folder, f) for f in frame_files]

    first_frame = cv2.imread(frame_paths[0])
    h, w, _ = first_frame.shape
    video_path = os.path.join(save_folder, "trajectory_video.mp4")
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for path in frame_paths:
        frame = cv2.imread(path)
        writer.write(frame)

    writer.release()
    print(f"\n✅ 2D Car animation saved at: {video_path}\n✅ Frames stored at: {frames_folder}")

if __name__ == "__main__":
    traj_8 = [
        [0, 0, 0],
        [1, 2, np.pi/4],
        [2, 3, np.pi/2],
        [3, 5, np.pi],
        [4, 6, -np.pi/2],
        [6, 5, -3*np.pi/4],
        [7, 3, -np.pi/2],
        [8, 0, np.pi/2],
    ]

    # Put a car.png file in simulation_output/ folder before running
    simulate_car(traj_8, save_folder="simulation_output", fps=20, car_img_path="car.png")
