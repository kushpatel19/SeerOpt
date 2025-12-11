"""
Script to calculate PDM scores for trajectories from GTSAM.

This script:
1. Loads trajectories from gtsam.trajectories directory (.npy files)
2. Calculates PDM scores for each trajectory
3. Saves results to CSV in gtsam directory
"""

import os
import sys
from pathlib import Path

# Check if running in correct environment
def check_environment():
    """Check if required packages are available."""
    missing_packages = []
    
    try:
        import numpy
    except ImportError:
        missing_packages.append("numpy")
    
    try:
        import pandas
    except ImportError:
        missing_packages.append("pandas")
    
    try:
        import hydra
    except ImportError:
        missing_packages.append("hydra-core")
    
    if missing_packages:
        print("=" * 60)
        print("❌ ERROR: Missing required packages!")
        print("=" * 60)
        print(f"Missing: {', '.join(missing_packages)}")
        print()
        print("Please activate the conda environment first:")
        print("  conda activate navsim_seerdrive")
        print()
        print("Or install missing packages:")
        print(f"  pip install {' '.join(missing_packages)}")
        print("=" * 60)
        sys.exit(1)
    
    # Check if we're likely in the right environment
    python_path = sys.executable
    if "navsim_seerdrive" not in python_path and "conda" not in python_path.lower():
        print("⚠️  Warning: You may not be in the correct conda environment.")
        print(f"   Python path: {python_path}")
        print("   Recommended: conda activate navsim_seerdrive")
        print()

# Run environment check first
check_environment()

import numpy as np
import pandas as pd
from tqdm import tqdm
import traceback
import lzma
import pickle
from typing import Dict, List, Any
from dataclasses import asdict
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set environment variables if not already set
if "NAVSIM_EXP_ROOT" not in os.environ:
    os.environ["NAVSIM_EXP_ROOT"] = str(PROJECT_ROOT / "exp")
if "OPENSCENE_DATA_ROOT" not in os.environ:
    os.environ["OPENSCENE_DATA_ROOT"] = "/home/harsh/dataset"
if "NUPLAN_MAPS_ROOT" not in os.environ:
    os.environ["NUPLAN_MAPS_ROOT"] = "/home/harsh/navsim/download/maps"
if "NUPLAN_MAP_VERSION" not in os.environ:
    os.environ["NUPLAN_MAP_VERSION"] = "nuplan-maps-v1.0"

import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.common.dataclasses import Trajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

# Default configuration path (relative to navsim/planning/script)
# Config path relative to navsim/planning/script directory
CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_scoring_parameters"  # YAML filename without .yaml extension

def load_gtsam_trajectories(trajectories_dir: Path) -> Dict[str, np.ndarray]:
    """
    Load all trajectory .npy files from gtsam.trajectories directory.
    Also checks converted_trajectories subfolder for converted files.
    
    Args:
        trajectories_dir: Path to directory containing .npy trajectory files
        
    Returns:
        Dictionary mapping token (filename without .npy) to trajectory array
    """
    trajectories = {}
    
    if not trajectories_dir.exists():
        raise FileNotFoundError(f"Trajectories directory not found: {trajectories_dir}")
    
    # Check main folder and converted_trajectories subfolder
    main_files = list(trajectories_dir.glob("*.npy"))
    converted_dir = trajectories_dir / "converted_trajectories"
    
    if converted_dir.exists():
        # Check for both .npy and .pkl files in converted folder
        converted_npy = list(converted_dir.glob("*.npy"))
        converted_pkl = list(converted_dir.glob("*.pkl"))
        converted_files = converted_pkl + converted_npy  # Prefer .pkl files
        print(f"Found {len(main_files)} files in {trajectories_dir}")
        print(f"Found {len(converted_files)} files in {converted_dir}")
        # Prefer converted files over original files (they're more compatible)
        # Add converted files first, then original files (will skip if token already exists)
        npy_files = converted_files + main_files
    else:
        npy_files = main_files
        print(f"Found {len(npy_files)} trajectory files in {trajectories_dir}")
    
    if not npy_files:
        raise ValueError(f"No .npy files found in {trajectories_dir} or {converted_dir}")
    
    for npy_file in npy_files:
        token = npy_file.stem  # filename without .npy or .pkl extension
        
        # Skip if we already loaded this token (prefer converted files)
        if token in trajectories:
            continue
        
        try:
            # Try loading with different numpy compatibility options
            traj_data = None
            errors = []
            
            # Method 1: Try pickle first (for converted .pkl files)
            import pickle
            if npy_file.suffix == '.pkl':
                try:
                    with open(npy_file, 'rb') as f:
                        traj_data = pickle.load(f)
                        # If it's a dict (converted format), we're good
                        if isinstance(traj_data, dict):
                            pass  # Keep as dict
                except Exception as e:
                    errors.append(f"pickle.load (.pkl): {str(e)}")
            
            # Method 2: Try numpy.load with allow_pickle (for .npy files)
            if traj_data is None:
                try:
                    traj_data = np.load(npy_file, allow_pickle=True)
                except Exception as e:
                    errors.append(f"numpy.load: {str(e)}")
                    
                    # If numpy.load fails, try pickle as fallback
                    try:
                        with open(npy_file, 'rb') as f:
                            traj_data = pickle.load(f)
                    except Exception as e2:
                        errors.append(f"pickle.load (fallback): {str(e2)}")
            
            # Method 3: Try with pickle protocol 4 (for Python 3.8+ compatibility)
            if traj_data is None:
                import pickle
                try:
                    with open(npy_file, 'rb') as f:
                        # Try with protocol 4 which is more compatible
                        traj_data = pickle.load(f)
                except Exception as e:
                    errors.append(f"pickle.load (protocol 4): {str(e)}")
            
            if traj_data is None:
                # Skip files that can't be loaded instead of crashing
                print(f"⚠️  Skipping {npy_file.name}: Cannot load (version mismatch)")
                print(f"   Errors: {'; '.join(errors)}")
                print(f"   💡 Solution: Convert using Python 3.13:")
                print(f"      conda deactivate")
                print(f"      python trajectory_utils/convert_numpy_file.py {npy_file}")
                print(f"      conda activate navsim_seerdrive")
                print()
                continue
            
            # Handle different formats:
            # 1. If it's a 0-d numpy array containing a dict, extract the dict
            if isinstance(traj_data, np.ndarray) and traj_data.ndim == 0:
                traj_data = traj_data.item()
            
            # 2. If it's a dictionary (like from evaluation), extract trajectories
            if isinstance(traj_data, dict):
                # The dict contains scene tokens as keys, each with trajectory data
                # We need to process all trajectories in the dict
                # Use the scene_token directly as the key (not prefixed with filename)
                # This matches the metric cache token format
                for scene_token, scene_traj in traj_data.items():
                    # Use scene_token directly (it matches metric cache tokens)
                    if scene_token in trajectories:
                        continue  # Skip if already loaded
                    
                    # Extract trajectory array (may be list or numpy array)
                    if isinstance(scene_traj, (list, tuple)):
                        # Convert list to numpy array
                        traj = np.array(scene_traj, dtype=np.float32)
                    elif isinstance(scene_traj, np.ndarray):
                        traj = scene_traj
                    elif isinstance(scene_traj, dict):
                        # If it's nested, try to extract the actual trajectory
                        traj = list(scene_traj.values())[0] if scene_traj else None
                        if traj is not None:
                            traj = np.array(traj, dtype=np.float32) if not isinstance(traj, np.ndarray) else traj
                    else:
                        traj = np.array(scene_traj, dtype=np.float32)
                    
                    if traj is not None:
                        # Convert to numpy array if not already
                        if not isinstance(traj, np.ndarray):
                            traj = np.array(traj, dtype=np.float32)
                        
                        # Ensure shape is (num_poses, 3)
                        if traj.ndim == 1:
                            traj = traj.reshape(-1, 3)
                        elif traj.ndim == 2 and traj.shape[1] != 3:
                            print(f"⚠️  Skipping {scene_token_key}: Expected shape (N, 3), got {traj.shape}")
                            continue
                        
                        trajectories[scene_token] = traj.astype(np.float32)
                continue  # Skip to next file after processing all scenes
            else:
                # Direct numpy array
                traj = traj_data
            
                # Convert to numpy array if not already
                if not isinstance(traj, np.ndarray):
                    traj = np.array(traj)
                
                # Ensure shape is (num_poses, 3)
                if traj.ndim == 1:
                    traj = traj.reshape(-1, 3)
                elif traj.ndim == 2 and traj.shape[1] != 3:
                    raise ValueError(f"Expected trajectory shape (N, 3), got {traj.shape}")
                
                trajectories[token] = traj.astype(np.float32)
            
        except Exception as e:
            print(f"Warning: Failed to load {npy_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return trajectories


def create_trajectory_object(poses: np.ndarray, time_horizon: float = 4.0, interval_length: float = 0.1) -> Trajectory:
    """
    Create a Trajectory dataclass from numpy array of poses.
    
    Args:
        poses: numpy array of shape (num_poses, 3) with [x, y, heading]
        time_horizon: Time horizon for trajectory sampling (default 4.0s)
        interval_length: Interval length for trajectory sampling (default 0.1s)
        
    Returns:
        Trajectory object
    """
    # The config expects num_poses=40, interval_length=0.1
    # Calculate time_horizon from num_poses: time_horizon = (num_poses - 1) * interval_length
    # For num_poses=40: time_horizon = 39 * 0.1 = 3.9
    # But we'll use the config's num_poses directly
    expected_num_poses = 40  # From config
    actual_num_poses = poses.shape[0]
    
    # If poses don't match expected number, interpolate to match
    if actual_num_poses != expected_num_poses:
        # Interpolate to match expected number of poses
        old_indices = np.linspace(0, 1, actual_num_poses)
        new_indices = np.linspace(0, 1, expected_num_poses)
        
        poses_interp = np.zeros((expected_num_poses, 3))
        for i in range(3):
            # Linear interpolation using numpy
            poses_interp[:, i] = np.interp(new_indices, old_indices, poses[:, i])
        
        # For heading, ensure it's in valid range [-pi, pi]
        poses_interp[:, 2] = np.arctan2(np.sin(poses_interp[:, 2]), np.cos(poses_interp[:, 2]))
        poses = poses_interp.astype(np.float32)
    
    num_poses = expected_num_poses  # Use expected num_poses
    # Calculate time_horizon from num_poses and interval_length
    # TrajectorySampling validation: time_horizon should equal num_poses * interval_length
    time_horizon = num_poses * interval_length
    
    # Remove duplicate interpolation code
    if False:  # This block is now redundant
        # Interpolate to match expected number of poses using numpy
        old_indices = np.linspace(0, 1, len(poses))
        new_indices = np.linspace(0, 1, num_poses)
        
        poses_interp = np.zeros((num_poses, 3))
        for i in range(3):
            # Linear interpolation using numpy
            poses_interp[:, i] = np.interp(new_indices, old_indices, poses[:, i])
        
        # For heading, ensure it's in valid range [-pi, pi]
        poses_interp[:, 2] = np.arctan2(np.sin(poses_interp[:, 2]), np.cos(poses_interp[:, 2]))
        poses = poses_interp.astype(np.float32)
    
    trajectory_sampling = TrajectorySampling(
        num_poses=num_poses,
        time_horizon=time_horizon,
        interval_length=interval_length
    )
    
    return Trajectory(poses=poses, trajectory_sampling=trajectory_sampling)


def calculate_pdm_scores(
    trajectories: Dict[str, np.ndarray],
    metric_cache_path: Path,
    config_path: str = CONFIG_PATH,
    config_name: str = CONFIG_NAME
) -> List[Dict[str, Any]]:
    """
    Calculate PDM scores for all trajectories.
    
    Args:
        trajectories: Dictionary mapping token to trajectory array
        metric_cache_path: Path to metric cache directory
        config_path: Path to Hydra config
        config_name: Name of Hydra config
        
    Returns:
        List of score dictionaries, one per trajectory
    """
    # Initialize Hydra - config_path should be relative to navsim/planning/script
    # Similar to how run_pdm_score.py does it (uses "config/pdm_scoring")
    script_dir = PROJECT_ROOT / "navsim/planning/script"
    config_path_abs = script_dir / CONFIG_PATH
    
    if not config_path_abs.exists():
        raise FileNotFoundError(
            f"Config directory not found: {config_path_abs}\n"
            f"Expected location: {PROJECT_ROOT}/navsim/planning/script/config/pdm_scoring"
        )
    
    # Change to navsim/planning/script directory for Hydra to find configs correctly
    # Hydra's initialize() uses the current working directory as base for relative paths
    import os
    original_cwd = os.getcwd()
    
    try:
        # Change to script directory (Hydra looks for configs relative to CWD)
        os.chdir(str(script_dir))
        actual_cwd = os.getcwd()
        
        # Verify the config exists from this directory
        config_check = Path(CONFIG_PATH)
        if not config_check.exists():
            raise FileNotFoundError(
                f"Config not found from {actual_cwd}: {CONFIG_PATH}\n"
                f"Absolute path: {config_path_abs}\n"
                f"Script dir: {script_dir}\n"
                f"Files in config/: {list(Path('config').iterdir()) if Path('config').exists() else 'config/ does not exist'}"
            )
        
        # Clear any existing Hydra state first (in case of previous initialization)
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        
        # Initialize Hydra - use absolute path to config directory
        # Hydra seems to use script location as base, so we'll use absolute path
        config_dir_abs = str(config_path_abs.resolve())
        config_name_without_ext = CONFIG_NAME
        
        with hydra.initialize_config_dir(config_dir=config_dir_abs, version_base=None):
            cfg = hydra.compose(config_name=config_name_without_ext)
    finally:
        # Always restore original directory
        os.chdir(original_cwd)
    
    # Initialize simulator and scorer
    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    
    # Load metric cache
    metric_cache_loader = MetricCacheLoader(metric_cache_path)
    
    # Get tokens that have both trajectory and metric cache
    available_tokens = list(set(trajectories.keys()) & set(metric_cache_loader.tokens))
    missing_tokens = set(trajectories.keys()) - set(metric_cache_loader.tokens)
    
    if missing_tokens:
        print(f"Warning: {len(missing_tokens)} trajectories don't have metric cache. Skipping them.")
        print(f"Missing tokens: {list(missing_tokens)[:10]}...")  # Show first 10
    
    if not available_tokens:
        raise ValueError("No tokens found with both trajectory and metric cache!")
    
    print(f"Calculating PDM scores for {len(available_tokens)} trajectories...")
    
    score_rows = []
    
    for token in tqdm(available_tokens, desc="Processing trajectories"):
        score_row: Dict[str, Any] = {"token": token, "valid": True}
        
        try:
            # Load metric cache for this token
            metric_cache_file = metric_cache_loader.metric_cache_paths[token]
            with lzma.open(metric_cache_file, "rb") as f:
                metric_cache = pickle.load(f)
            
            # Get trajectory array
            traj_poses = trajectories[token]
            
            # Create Trajectory object
            model_trajectory = create_trajectory_object(
                traj_poses,
                time_horizon=simulator.proposal_sampling.time_horizon,
                interval_length=simulator.proposal_sampling.interval_length
            )
            
            # Calculate PDM score
            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=model_trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            
            # Update score row with results
            score_row.update(asdict(pdm_result))
            
        except Exception as e:
            print(f"\nError processing token {token}: {e}")
            traceback.print_exc()
            score_row["valid"] = False
        
        score_rows.append(score_row)
    
    return score_rows


def save_results(score_rows: List[Dict[str, Any]], output_dir: Path):
    """
    Save PDM scores to CSV file.
    
    Args:
        score_rows: List of score dictionaries
        output_dir: Directory to save CSV file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame
    df = pd.DataFrame(score_rows)
    
    # Calculate statistics
    num_successful = df["valid"].sum()
    num_failed = len(df) - num_successful
    
    # Calculate average row
    if num_successful > 0:
        average_row = df[df["valid"]].drop(columns=["token", "valid"]).mean(skipna=True)
        average_row["token"] = "average"
        average_row["valid"] = True
        df = pd.concat([df, pd.DataFrame([average_row])], ignore_index=True)
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    csv_path = output_dir / f"gtsam_pdm_scores_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"\n✅ Results saved to: {csv_path}")
    print(f"   Successful: {num_successful}, Failed: {num_failed}")
    
    if num_successful > 0 and 'score' in df.columns:
        avg_score = df[df["valid"]]["score"].mean()
        print(f"   Average PDM Score: {avg_score:.4f}")
    
    return csv_path


def main():
    """Main function to run PDM scoring for GTSAM trajectories."""
    
    # Set up paths
    project_root = Path(__file__).parent.parent.absolute()
    trajectories_dir = project_root / "gtsam" / "trajectories"
    converted_dir = trajectories_dir / "converted_trajectories"
    metric_cache_path = Path(os.environ.get("NAVSIM_EXP_ROOT", project_root / "exp")) / "metric_cache"
    output_dir = project_root / "gtsam"
    
    print("=" * 60)
    print("GTSAM Trajectory PDM Scoring")
    print("=" * 60)
    print(f"Trajectories directory: {trajectories_dir}")
    if converted_dir.exists():
        print(f"Converted trajectories: {converted_dir} (will be preferred)")
    print(f"Metric cache path: {metric_cache_path}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Check if directories exist
    if not trajectories_dir.exists():
        print(f"❌ Error: Trajectories directory not found: {trajectories_dir}")
        print(f"   Please create the directory and add .npy trajectory files.")
        return
    
    if not metric_cache_path.exists():
        print(f"❌ Error: Metric cache directory not found: {metric_cache_path}")
        print(f"   Please run metric caching first: bash scripts/evaluation/run_metric_caching.sh")
        return
    
    # Load trajectories
    try:
        trajectories = load_gtsam_trajectories(trajectories_dir)
        print(f"✅ Loaded {len(trajectories)} trajectories")
    except Exception as e:
        print(f"❌ Error loading trajectories: {e}")
        traceback.print_exc()
        return
    
    # Calculate PDM scores
    try:
        score_rows = calculate_pdm_scores(trajectories, metric_cache_path)
    except Exception as e:
        print(f"❌ Error calculating PDM scores: {e}")
        traceback.print_exc()
        return
    
    # Save results
    try:
        csv_path = save_results(score_rows, output_dir)
        print(f"\n✅ Complete! Results saved to: {csv_path}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()