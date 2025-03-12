import os
import sys

def rename_files(directory):
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return
    for filename in os.listdir(directory):
        if filename.endswith(".xyz"):
            old_path = os.path.join(directory, filename)
            new_filename = filename.replace(".xyz", ".dump")
            new_path = os.path.join(directory, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rename_xyz_to_dump.py <directory>")
        sys.exit(1)
    target_directory = sys.argv[1]
    rename_files(target_directory)
