import os
import csv

def fix_tumor_areas_csv(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # Read the contents of the CSV file
    tumor_areas = []
    with open(file_path, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        for row in csvreader:
            tumor_areas.append(row)

    # Fill the "Slice Number" column with numbers from 0 to the length of "Tumor Area"
    for i in range(len(tumor_areas)):
        tumor_areas[i][1] = str(i)

    # Write the corrected data back to the CSV file
    with open(file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)  # Write the original header
        csvwriter.writerows(tumor_areas)

    print(f"Successfully fixed tumor_areas.csv in: {file_path}")

# Example usage:
# Replace 'path_to_npy_folder' with the path to the folder containing all the patient NPY folders
def fix_all_tumor_areas_csv(path_to_npy_folder):
    for root, _, files in os.walk(path_to_npy_folder):
        for file in files:
            if file == 'tumor_areas.csv':
                tumor_areas_csv_path = os.path.join(root, file)
                fix_tumor_areas_csv(tumor_areas_csv_path)

fix_all_tumor_areas_csv('/media/adamdiakite/LaCie/CT-TEP_ICI/NPY')
