import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ANSI escape codes for vibrant text formatting
class TextStyle:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def convert_image_to_grayscale(input_file, output_file):
    """
    Convert a single image to grayscale and save it.
    """
    img = Image.open(input_file).convert("L")
    img.save(output_file)

def standardize_image_name(filename):
    """
    Standardize the image name format.
    """
    name, ext = os.path.splitext(filename)
    standardized_name = f"{name.lower().replace(' ', '_')}{ext.lower()}"
    return standardized_name

def process_directory(input_dir, output_dir):
    """
    Convert all images in the input directory to grayscale and save them to the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List of image files
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('jpg', 'jpeg', 'png'))]

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        # Create a list of tasks
        tasks = []
        for filename in image_files:
            input_file = os.path.join(input_dir, filename)
            standardized_name = standardize_image_name(filename)
            output_file = os.path.join(output_dir, standardized_name)
            tasks.append(executor.submit(convert_image_to_grayscale, input_file, output_file))

        # Add progress bar using tqdm
        for _ in tqdm(tasks, desc=f"Processing {os.path.basename(input_dir)}"):
            # Wait for the task to complete
            _.result()

def preprocess_dataset(dataset_name, dataset_types):
    """
    Preprocess the dataset by converting images to grayscale in specified folders.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', dataset_name))

    if not os.path.exists(base_dir):
        print(f"{TextStyle.FAIL}Base data directory not found: {base_dir}{TextStyle.ENDC}")
        return

    available_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    while not set(dataset_types).issubset(set(available_dirs)):
        print(f"{TextStyle.WARNING}Available directories in {dataset_name}: {', '.join(available_dirs)}{TextStyle.ENDC}")
        dataset_types = input(f"{TextStyle.BOLD}Enter the dataset types to process from the available directories (separated by comma) or type 'quit' to exit: {TextStyle.ENDC}").strip().lower().split(',')
        if 'quit' in dataset_types:
            print(f"{TextStyle.OKBLUE}Exiting...{TextStyle.ENDC}")
            return
        dataset_types = [dt.strip() for dt in dataset_types if dt.strip()]

    for dataset_type in dataset_types:
        dataset_dir = os.path.join(base_dir, dataset_type)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        input_dir = os.path.join(dataset_dir, 'original')
        grayscale_dir = os.path.join(dataset_dir, 'grayscale')

        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
        if not os.path.exists(grayscale_dir):
            os.makedirs(grayscale_dir)

        process_directory(input_dir, grayscale_dir)
        print(f"{TextStyle.OKGREEN}Grayscale conversion completed for {dataset_type} dataset in {dataset_name}{TextStyle.ENDC}")

# Example usage
if __name__ == "__main__":
    while True:
        dataset_name = input(f"{TextStyle.BOLD}Enter the dataset name (or type 'quit' to exit): {TextStyle.ENDC}").strip().lower()
        if dataset_name == 'quit':
            print(f"{TextStyle.OKBLUE}Exiting...{TextStyle.ENDC}")
            break
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', dataset_name))
        if not os.path.exists(base_dir):
            print(f"{TextStyle.FAIL}Base data directory not found: {base_dir}{TextStyle.ENDC}")
            available_datasets = [d for d in os.listdir(os.path.join(os.path.dirname(__file__), '..', 'data')) if os.path.isdir(os.path.join(os.path.dirname(__file__), '..', 'data', d))]
            print(f"{TextStyle.WARNING}Available datasets: {', '.join(available_datasets)}{TextStyle.ENDC}")
        else:
            dataset_types = input(f"{TextStyle.BOLD}Enter the dataset types to process (train, val, or both separated by comma): {TextStyle.ENDC}").strip().lower().split(',')
            # Remove any extra spaces around dataset types
            dataset_types = [dt.strip() for dt in dataset_types if dt.strip()]
            preprocess_dataset(dataset_name, dataset_types)
            break