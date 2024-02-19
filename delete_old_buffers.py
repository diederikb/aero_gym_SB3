import os
import re
import sys

"""

Python script to remove all replay buffers except for the last generated one.

"""

def find_and_delete_buffers(directory):
    pattern_pkl = re.compile(r'rl_model_replay_buffer_(\d+)_steps.pkl')

    files_dict_pkl = {}

    # Iterate through files in the directory
    for filename in os.listdir(directory):
        match_pkl = pattern_pkl.match(filename)
        # match_zip = pattern_zip.match(filename)

        if match_pkl:
            number = int(match_pkl.group(1))
            files_dict_pkl[number] = filename

    # Determine the file with the highest value for NUMBER for pkl files
    if files_dict_pkl:
        max_number_pkl = max(files_dict_pkl.keys())
        file_to_keep_pkl = files_dict_pkl[max_number_pkl]
        print(f"PKL File to keep: {file_to_keep_pkl}")
    else:
        print("No matching pkl files found.")
        sys.exit(1)

    # Ask the user for confirmation
    confirmation = input("Do you want to delete all other files? (yes/no): ").lower()

    if confirmation == 'yes':
        # Delete all other pkl files
        for number, filename in files_dict_pkl.items():
            if filename != file_to_keep_pkl:
                file_path = os.path.join(directory, filename)
                os.remove(file_path)

        print(f"Files deleted except for {file_to_keep_pkl}")
    else:
        print("Deletion aborted.")

if __name__ == "__main__":
    # Check if the target directory is provided as a command line argument
    if len(sys.argv) < 2:
        print("Usage: python delete_old_buffers.py /path/to/your/target/directory")
        sys.exit(1)

    # Use glob to expand wildcard * in the directory path
    directories = sys.argv[1:]

    if not directories:
        print(f"No directories found for: {directories}")
        sys.exit(1)

    for directory in directories:
        # Call the function for each directory
        find_and_delete_buffers(directory)
