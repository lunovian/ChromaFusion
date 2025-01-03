import os
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Define the source and destination directories
source_dir = 'C:/Users/nguye/.cache/kagglehub/datasets/benjaminkz/places365/versions/1'
destination_dir = 'C:/Users/nguye/Documents/Github/chroma-fusion/data/places365'

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Get the list of files in the source directory
files = os.listdir(source_dir)

def move_file(filename):
    source_file = os.path.join(source_dir, filename)
    destination_file = os.path.join(destination_dir, filename)
    shutil.move(source_file, destination_file)

# Move the files from the source directory to the destination directory with a progress bar
with ThreadPoolExecutor(max_workers=8) as executor:
    list(tqdm(executor.map(move_file, files), total=len(files), desc="Moving files", unit="file"))

print("Files moved to:", destination_dir)