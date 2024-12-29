import os
import cv2
import numpy as np
import concurrent.futures

def is_black_and_white(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return False
    if np.array_equal(image[:, :, 0], image[:, :, 1]) and np.array_equal(image[:, :, 0], image[:, :, 2]):
        return True
    return False

def process_image(file_path):
    try:
        if is_black_and_white(file_path):
            os.remove(file_path)
            return f"Deleted: {file_path}"
        return None
    except Exception as e:
        return f"Error processing {file_path}: {e}"

def delete_black_and_white_images(directory):
    all_files = []
    for root, _, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            if os.path.isfile(file_path):
                all_files.append(file_path)
    
    initial_count = len(all_files)
    print(f"Initial file count in {directory}: {initial_count}")
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_image, all_files))
        for result in results:
            if result:
                print(result)
    
    final_files = [file for file in all_files if os.path.isfile(file)]
    final_count = len(final_files)
    print(f"Final file count in {directory}: {final_count}")

directories = [
    r'C:\Users\Admin\Documents\REColor\input\finetune\history',
    r'C:\Users\Admin\Documents\REColor\input\finetune\human_face',
]

for directory in directories:
    delete_black_and_white_images(directory)
