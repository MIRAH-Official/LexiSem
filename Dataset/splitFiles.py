
import os
import json
import shutil

def move_files(file_list, source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for file in file_list:
        shutil.move(os.path.join(source_dir, file), os.path.join(target_dir, file))

def categorize_files(dataset_path, processed_dir, unprocessed_dir1, unprocessed_dir2):
    unprocessed_files = []
    processed_files = []
    files = [f for f in os.listdir(dataset_path) if f.endswith('.json')]
    
    for filename in files:
        file_path = os.path.join(dataset_path, filename)
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
            if "segmented_article" in data:
                processed_files.append(filename)
            else:
                unprocessed_files.append(filename)
        except json.JSONDecodeError:
            print(f"Error reading JSON file: {filename}")

    if processed_files:
        move_files(processed_files, dataset_path, processed_dir)

    if unprocessed_files:
        half_point = len(unprocessed_files) // 2
        move_files(unprocessed_files[:half_point], dataset_path, unprocessed_dir1)
        move_files(unprocessed_files[half_point:], dataset_path, unprocessed_dir2)

def main():
    dataset_path = 'train'
    processed_dir = 'processed'
    unprocessed_dir1 = 'unprocessed_part11'
    unprocessed_dir2 = 'unprocessed_part22'
    
    categorize_files(dataset_path, processed_dir, unprocessed_dir1, unprocessed_dir2)
    print(f"Files have been moved to processed, {unprocessed_dir1}, and {unprocessed_dir2}")

if __name__ == "__main__":
    main()

