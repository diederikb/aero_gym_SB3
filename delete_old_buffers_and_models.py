import os
import re
import sys

def find_and_delete_files(directory):
    pattern_pkl = re.compile(r'rl_model_replay_buffer_(\d+)_steps.pkl')
    pattern_zip = re.compile(r'rl_model_(\d+)_steps.zip')

    files_dict_pkl = {}
    files_dict_zip = {}

    # Iterate through files in the directory
    for filename in os.listdir(directory):
        match_pkl = pattern_pkl.match(filename)
        match_zip = pattern_zip.match(filename)

        if match_pkl:
            number = int(match_pkl.group(1))
            if number % 10000 == 0:
                files_dict_pkl[number] = filename

        elif match_zip:
            number = int(match_zip.group(1))
            if number % 10000 == 0:
                files_dict_zip[number] = filename

    # Determine the file with the highest value for NUMBER for pkl files
    if files_dict_pkl:
        max_number_pkl = max(files_dict_pkl.keys())
        file_to_keep_pkl = files_dict_pkl[max_number_pkl]
        print(f"PKL File to keep: {file_to_keep_pkl}")
    else:
        print("No matching pkl files found.")

    # Determine the file with the highest value for NUMBER for zip files
    if files_dict_zip:
        max_number_zip = max(files_dict_zip.keys())
        file_to_keep_zip = files_dict_zip[max_number_zip]
        print(f"ZIP File to keep: {file_to_keep_zip}")
    else:
        print("No matching zip files found.")

    # Ask the user for confirmation
    confirmation = input("Do you want to delete all other files? (yes/no): ").lower()

    if confirmation == 'yes':
        # Delete all other pkl files
        for number, filename in files_dict_pkl.items():
            if filename != file_to_keep_pkl:
                file_path = os.path.join(directory, filename)
                os.remove(file_path)

        # Delete all other zip files
        for number, filename in files_dict_zip.items():
            if filename != file_to_keep_zip:
                file_path = os.path.join(directory, filename)
                os.remove(file_path)

        print(f"Files deleted except for {file_to_keep_pkl} (PKL) and {file_to_keep_zip} (ZIP)")
    else:
        print("Deletion aborted.")

if __name__ == "__main__":
    # Check if the target directory is provided as a command line argument
    if len(sys.argv) != 2:
        print("Usage: python script.py /path/to/your/target/directory")
        sys.exit(1)

    target_directory = sys.argv[1]

    # Call the function
    find_and_delete_files(target_directory)
