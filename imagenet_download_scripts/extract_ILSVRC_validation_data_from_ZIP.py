import tarfile
from concurrent.futures import ThreadPoolExecutor
import os
import shutil

# Define the directory and file paths
test_directory = "imagenet/val"
file_to_move = "ILSVRC2012_img_val.tar"

# Create imagenet/train directory (including parent direcories if they don't exist)
os.makedirs(test_directory, exist_ok=True)

# Create file path to move tar into
move_test_tar_file_path = os.path.join(test_directory, os.path.basename(file_to_move))

# Move ILSVRC2012_img_train.tar to the imagenet/train directory
shutil.move(file_to_move, move_test_tar_file_path)

print(f"Moved {file_to_move} to {move_test_tar_file_path}")

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

# Define paths
extract_to_directory = "imagenet/val"

extract_tar_parallel(move_test_tar_file_path, extract_to=extract_to_directory)
