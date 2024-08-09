import tarfile
from concurrent.futures import ThreadPoolExecutor
import os
import shutil

# Define the directory and file paths
train_directory = "imagenet/train"
file_to_move = "ILSVRC2012_img_train.tar"

# Create imagenet/train directory (including parent direcories if they don't exist)
os.makedirs(train_directory, exist_ok=True)

# Create file path to move tar into
move_train_tar_file_path = os.path.join(train_directory, os.path.basename(file_to_move))

# Move ILSVRC2012_img_train.tar to the imagenet/train directory
shutil.move(file_to_move, move_train_tar_file_path)

print(f"Moved {file_to_move} to {move_train_tar_file_path}")

def extract_member(tar_path, member, extract_to="."):
    with tarfile.open(tar_path, 'r') as tar:
        tar.extract(member, path=extract_to)

def extract_tar_parallel(tar_path, extract_to=".", num_workers=32):
    with tarfile.open(tar_path, 'r') as tar:
        members = tar.getmembers()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for member in members:
            executor.submit(extract_member, tar_path, member, extract_to)

    # Remove the tar file after extraction
    os.remove(tar_path)
    print(f"Removed tar file: {tar_path}")

def extract_tar(tar_path, extract_to="."):
    """
    Extracts a tar file to a specified directory
    """
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=extract_to)
    print(f"Extracted {tar_path} to {extract_to}")

def extract_nested_tars(base_dir, num_workers=32):
    """
    Finds and extracts all tar files in a base directory, extracts their contents
    into corresponding directories, and deletes the tar files
    """
    # Find all tar files within the base directory
    nested_tar_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".tar"):
                nested_tar_files.append(os.path.join(root, file))
    
    def process_nested_tar(nested_tar_path):
        """
        Process a single nested tar file: extract it and then delete it.
        """
        # Create a directory for the extracted contents
        extract_dir = nested_tar_path[:-4] # Remove the ".tar" extension
        os.makedirs(extract_dir, exist_ok=True)
        # Extract the nested tar file
        extract_tar(nested_tar_path, extract_to=extract_dir)
        # Remove the tar file after extraction
        os.remove(nested_tar_path)
        print(f"Removed {nested_tar_path}")
    
    # Use ThreadPoolExecutor to process nested tar files in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(process_nested_tar, nested_tar_files)

# Define paths
extract_to_directory = "imagenet/train"

extract_tar_parallel(move_train_tar_file_path, extract_to=extract_to_directory)
extract_nested_tars(extract_to_directory)
