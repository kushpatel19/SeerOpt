"""
Helper script to convert numpy files with version compatibility issues.
Can convert a single file or all .npy files in a folder.
"""

import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

def convert_numpy_file(input_file, output_file=None):
    """Convert numpy file to compatible format."""
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return False
    
    if output_file is None:
        output_file = input_path.parent / f"{input_path.stem}_converted{input_path.suffix}"
    
    output_path = Path(output_file)
    
    # Try to load with different methods
    data = None
    errors = []
    
    # Method 1: Try with numpy.load
    try:
        data = np.load(str(input_path), allow_pickle=True)
    except Exception as e:
        error_msg = str(e)
        errors.append(f"numpy.load: {error_msg}")
        # If it's a numpy._core error, try alternative methods
        if "numpy._core" in error_msg or "_core" in error_msg or "No module named" in error_msg:
            pass  # Will try pickle below
        else:
            # Re-raise if it's a different error
            raise
    
    # Method 2: Try with pickle (for version compatibility)
    if data is None:
        try:
            import pickle
            with open(input_path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            errors.append(f"pickle.load: {str(e)}")
            # If both methods fail, suggest using Python 3.13
            if "STACK_GLOBAL" in str(e) or "numpy._core" in str(e):
                print(f"❌ Failed to load {input_path.name}: {'; '.join(errors)}")
                print()
                print("💡 SOLUTION: This file was created with Python 3.13/NumPy 2.x")
                print("   You need to convert it using Python 3.13 first:")
                print()
                print("   1. Deactivate current environment:")
                print("      conda deactivate")
                print()
                print("   2. Run conversion in base Python 3.13:")
                print(f"      python trajectory_utils/convert_numpy_file.py {input_path}")
                print()
                print("   3. Then use the converted file in navsim_seerdrive environment")
                return False
            print(f"❌ Failed to load {input_path.name}: {'; '.join(errors)}")
            return False
    
    # Save in compatible format
    # The data structure is: 0-d numpy array containing a dict
    # We need to preserve this structure but save it in a compatible way
    try:
        # If data is a 0-d array containing a dict, extract the dict
        if isinstance(data, np.ndarray) and data.ndim == 0 and isinstance(data.item(), dict):
            data_dict = data.item()
            
            # Convert all numpy arrays in the dict to plain Python lists
            # This is necessary because NumPy 2.x arrays reference numpy._core
            # By converting to lists, we avoid pickle issues entirely
            def convert_numpy_arrays(obj):
                """Recursively convert numpy arrays to plain Python lists."""
                if isinstance(obj, np.ndarray):
                    # Convert to plain Python list (no numpy references)
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_arrays(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return type(obj)(convert_numpy_arrays(item) for item in obj)
                else:
                    return obj
            
            # Convert all arrays in the dict to lists
            converted_dict = convert_numpy_arrays(data_dict)
            
            # Save the dict directly using pickle with protocol 4 (compatible with Python 3.4+)
            # Save as .pkl file for better compatibility
            import pickle
            pkl_path = output_path.with_suffix('.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump(converted_dict, f, protocol=4)
            print(f"   Also saved as: {pkl_path.name}")
            return True
        
        # If data is already a dict, save it directly
        if isinstance(data, dict):
            import pickle
            with open(str(output_path), 'wb') as f:
                pickle.dump(data, f, protocol=4)
            return True
        
        # For regular numpy arrays, try to save without pickle first
        if isinstance(data, np.ndarray):
            try:
                np.save(str(output_path), data, allow_pickle=False)
                return True
            except (ValueError, TypeError) as e:
                # If it fails (e.g., object arrays), use pickle
                if "Object arrays" in str(e) or "cannot be saved" in str(e):
                    import pickle
                    with open(str(output_path), 'wb') as f:
                        pickle.dump(data, f, protocol=4)
                    return True
                else:
                    raise
        
        # Fallback: convert to numpy array
        data_array = np.array(data)
        try:
            np.save(str(output_path), data_array, allow_pickle=False)
            return True
        except (ValueError, TypeError):
            import pickle
            with open(str(output_path), 'wb') as f:
                pickle.dump(data, f, protocol=4)
            return True
    except Exception as e:
        print(f"❌ Failed to save {output_path.name}: {e}")
        return False


def convert_folder(input_folder, output_folder=None, overwrite=False):
    """
    Convert all .npy files in a folder.
    
    Args:
        input_folder: Path to folder containing .npy files
        output_folder: Path to output folder (default: converted_trajectories subfolder)
        overwrite: If True, overwrite existing files. If False, skip already converted files.
    """
    input_path = Path(input_folder)
    
    if not input_path.exists():
        print(f"❌ Error: Folder not found: {input_path}")
        return False
    
    if not input_path.is_dir():
        print(f"❌ Error: Not a directory: {input_path}")
        return False
    
    # Find all .npy files
    npy_files = list(input_path.glob("*.npy"))
    
    if not npy_files:
        print(f"⚠️  No .npy files found in {input_path}")
        return False
    
    # Set up output folder
    if output_folder is None:
        # Default: save to converted_trajectories subfolder
        output_path = input_path / "converted_trajectories"
    else:
        output_path = Path(output_folder)
    
    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Found {len(npy_files)} .npy files in {input_path}")
    print(f"📁 Output folder: {output_path}")
    print()
    
    # Convert each file
    successful = 0
    failed = 0
    skipped = 0
    
    for npy_file in tqdm(npy_files, desc="Converting files"):
        # Determine output file path
        # Always save to output_path with same filename (no _converted suffix)
        output_file = output_path / npy_file.name
        
        # Skip if already exists and not overwriting
        if output_file.exists() and not overwrite:
            skipped += 1
            continue
        
        # Convert the file
        if convert_numpy_file(npy_file, output_file):
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print()
    print("=" * 60)
    print("Conversion Summary:")
    print(f"  ✅ Successful: {successful}")
    print(f"  ❌ Failed: {failed}")
    print(f"  ⏭️  Skipped: {skipped}")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single file: python convert_numpy_file.py <input_file.npy> [output_file.npy]")
        print("  Folder:      python convert_numpy_file.py <input_folder> [output_folder] [--overwrite]")
        print()
        print("Examples:")
        print("  python convert_numpy_file.py file.npy")
        print("  python convert_numpy_file.py gtsam/trajectories/")
        print("    -> Converts to: gtsam/trajectories/converted_trajectories/")
        print("  python convert_numpy_file.py gtsam/trajectories/ gtsam/trajectories_converted/")
        print("    -> Converts to: gtsam/trajectories_converted/")
        sys.exit(1)
    
    # Parse arguments, handling flags
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--") and not arg.startswith("-")]
    flags = [arg for arg in sys.argv[1:] if arg.startswith("--") or arg.startswith("-")]
    
    overwrite = "--overwrite" in flags or "-f" in flags or "--force" in flags
    
    if len(args) < 1:
        print("❌ Error: No input path provided")
        sys.exit(1)
    
    input_path = Path(args[0])
    output_path = Path(args[1]) if len(args) > 1 else None
    
    # Check if input is a file or folder
    if input_path.is_file():
        # Single file conversion
        success = convert_numpy_file(input_path, output_path)
        sys.exit(0 if success else 1)
    elif input_path.is_dir():
        # Folder conversion
        success = convert_folder(input_path, output_path, overwrite)
        sys.exit(0 if success else 1)
    else:
        print(f"❌ Error: Path not found: {input_path}")
        sys.exit(1)