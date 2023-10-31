import os
import shutil

import os
import pydicom
import dicom2nifti
import re

import pandas as pd


def find_and_remove_folders_without_dcm_nii(root_folder):
    # Step 1: Collect directories with both DCM and NII files
    def contains_dcm_and_nii(dirpath):
        dcm_found = False
        nii_found = False
        for _, _, filenames in os.walk(dirpath):
            for filename in filenames:
                if filename.lower().endswith(".dcm"):
                    dcm_found = True
                elif filename.lower().endswith(".nii"):
                    nii_found = True
                if dcm_found and nii_found:
                    return True
        return False

    for dirpath, _, _ in os.walk(root_folder, topdown=False):
        if not contains_dcm_and_nii(dirpath):
            print(f"Removing directory and its parent folders: {dirpath}")
            try:
                shutil.rmtree(dirpath)
            except OSError as e:
                print(f"Error deleting directory {dirpath}: {str(e)}")


def rename_folders(root_folder):
    for patient_folder in os.listdir(root_folder):
        patient_folder_path = os.path.join(root_folder, patient_folder)
        if os.path.isdir(patient_folder_path):
            folders_with_dcm_nii = []

            # Step 1: Identify folders with both DCM and NII files for each patient
            for dirpath, dirnames, filenames in os.walk(patient_folder_path):
                dcm_found = any(f.lower().endswith(".dcm") for f in filenames)
                nii_found = any(f.lower().endswith(".nii") for f in filenames)

                if dcm_found and nii_found:
                    folders_with_dcm_nii.append(dirpath)

            # Step 2: Rename folders and create .txt files for each patient
            scan_counter = {}
            for folder_path in folders_with_dcm_nii:
                base_name = "scan"
                count = scan_counter.get(patient_folder_path, 0)
                if count > 0:
                    base_name += f"_{count}"
                scan_counter[patient_folder_path] = count + 1

                new_folder_path = os.path.join(patient_folder_path, base_name)

                try:
                    os.rename(folder_path, new_folder_path)

                    # Create a .txt file with the previous folder name for record-keeping
                    txt_filename = os.path.basename(folder_path) + ".txt"
                    txt_filepath = os.path.join(new_folder_path, txt_filename)

                    with open(txt_filepath, "w") as txt_file:
                        txt_file.write("Folder was renamed to: " + base_name)

                except FileExistsError as e:
                    print(f"Folder {new_folder_path} already exists. Skipping.")


def delete_empty_folders(root_folder):
    for dirpath, dirnames, filenames in os.walk(root_folder, topdown=False):
        # Check if the current directory is empty (contains no files or subdirectories)
        if not dirnames and not filenames:
            print(f"Deleting empty folder: {dirpath}")
            try:
                os.rmdir(dirpath)  # Delete the empty folder
            except OSError as e:
                print(f"Error deleting folder {dirpath}: {str(e)}")


def create_txt_folder(root_folder):
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(".txt"):
                txt_filepath = os.path.join(dirpath, filename)
                folder_name = os.path.splitext(filename)[0]  # Remove the file extension

                # Create a folder with the same name as the txt file (if it doesn't exist)
                folder_path = os.path.join(dirpath, folder_name)
                os.makedirs(folder_path, exist_ok=True)

                # Move the txt file into the folder
                new_txt_filepath = os.path.join(folder_path, filename)
                shutil.move(txt_filepath, new_txt_filepath)


def convert_dicom_series_to_nifti_recursive(root_folder):
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(".dcm"):
                dicom_files = [os.path.join(dirpath, f) for f in filenames if f.lower().endswith(".dcm")]

                try:
                    dicom_series = [pydicom.dcmread(dicom_file) for dicom_file in dicom_files]
                    nifti_image = dicom2nifti.dicom_series_to_nifti(dicom_series)
                except dicom2nifti.exceptions.ConversionError as e:
                    print(f"Error converting DICOM series in {dirpath}: {str(e)}")
                    continue

                # Save the NIfTI file at the same location as the DICOM series
                nifti_filepath = os.path.join(dirpath, "scan.nii.gz")
                dicom2nifti.write_nifti(nifti_image, nifti_filepath)

#
# path = '/media/adamdiakite/LaCie/6-Lille_reformater_trier_contourer_VS'
# convert_dicom_series_to_nifti_recursive(path)


def list_folder_names_at_depth_2(root_dir):
    folder_names = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        depth = dirpath[len(root_dir):].count(os.sep)

        if depth == 2:
            for dirname in dirnames:
                folder_names.append(dirname)

    # Sort the folder names in ascending order
    folder_names.sort()

    return folder_names


# root_directory = "/home/adamdiakite/Documents/Patients_Vincent"
# folder_names_at_depth_2 = list_folder_names_at_depth_2(root_directory)
#
# for folder_name in folder_names_at_depth_2:
#     print(folder_name)
#
# print(len(folder_names_at_depth_2))


def group_folders_by_similarity(root_dir, min_common_length=5):
    folder_groups = {}

    for dirpath, dirnames, filenames in os.walk(root_dir):
        depth = dirpath[len(root_dir):].count(os.sep)

        if depth == 2:
            for dirname in dirnames:
                lowercase_dirname = dirname.lower()

                # Check if the lowercase dirname contains a single word of min_common_length letters
                words = re.findall(r'\b\w{5,}\b', lowercase_dirname)

                if len(words) == 1:
                    common_key = words[0]

                    # Extract the patient ID from the path (assuming it's part of the directory structure)
                    patient_id = os.path.basename(os.path.dirname(dirpath))[-7:]  # Extract last 7 characters

                    if common_key not in folder_groups:
                        folder_groups[common_key] = []

                    folder_groups[common_key].append(patient_id)

    return folder_groups


def list_files_with_key(root_directory, key):
    file_list = []

    for foldername, subfolders, filenames in os.walk(root_directory):
        for filename in filenames:
            if key in filename:
                file_list.append(os.path.join(foldername, filename))

    file_list.sort()  # Sort the list of files in ascending order

    return file_list


def rename_files_with_key(root_directory, key, new_name):
    renamed_files = []  # To store the renamed file paths

    for foldername, subfolders, filenames in os.walk(root_directory):
        for filename in filenames:
            if key in filename:
                old_path = os.path.join(foldername, filename)
                new_filename = new_name
                new_path = os.path.join(foldername, new_filename)
                os.rename(old_path, new_path)
                renamed_files.append((old_path, new_path))

    return renamed_files


def copy_folders_with_subfolder_structure(root_directory, key, destination_directory):
    # Iterate through folders in the root directory
    for foldername, subfolders, filenames in os.walk(root_directory):
        for subfolder in subfolders:
            if key.lower() in subfolder.lower():
                # If the 'key' is found in a subfolder's name, copy the entire directory structure
                source_folder = os.path.join(foldername, subfolder)
                destination_folder = os.path.join(destination_directory, os.path.relpath(source_folder, root_directory))

                # Copy the directory structure, overwriting existing files
                for dirpath, dirnames, files in os.walk(source_folder):
                    relative_dirpath = os.path.relpath(dirpath, source_folder)
                    new_dirpath = os.path.join(destination_folder, relative_dirpath)

                    if not os.path.exists(new_dirpath):
                        os.makedirs(new_dirpath)

                    for file in files:
                        source_file = os.path.join(dirpath, file)
                        destination_file = os.path.join(new_dirpath, file)
                        shutil.copy2(source_file, destination_file)


# # Example usage:
# root_dir = '/media/adamdiakite/LaCie/Patient_Vincent_Test'
# key_to_replace = 'ring'
# new_name = 'ring.nii.gz'
#
# renamed_files = rename_files_with_key(root_dir, key_to_replace, new_name)
#
# # Print the renamed file paths
# for old_path, new_path in renamed_files:
#     print(f'Renamed: {old_path} -> {new_path}')

# rename_files_with_key(root_dir, 'Ring', 'ring.nii.gz')
# print(list_files_with_key(root_dir, 'ring'))

import csv
import pydicom


def extract_dicom_info(dicom_folder):
    filter_type = "N/A"
    body_part = "N/A"
    convolution_kernel = "N/A"
    cine_rate = "N/A"
    contrast_bolus_agent = "N/A"

    for dicom_file in os.listdir(dicom_folder):
        dicom_file_path = os.path.join(dicom_folder, dicom_file)
        if os.path.isfile(dicom_file_path) and dicom_file_path.lower().endswith('.dcm'):
            try:
                dicom_data = pydicom.dcmread(dicom_file_path)
                filter_type = dicom_data.get("FilterType", "N/A")
                body_part = dicom_data.get("BodyPartExamined", "N/A")
                convolution_kernel = dicom_data.get("ConvolutionKernel", "N/A")
                cine_rate = dicom_data.get("CineRate", "N/A")
                contrast_bolus_agent = dicom_data.get("ContrastBolusAgent", "N/A")
                break  # Stop after finding information in the first DICOM file
            except Exception as e:
                pass  # Continue if there's an error with the DICOM file

    return filter_type, body_part, convolution_kernel, cine_rate, contrast_bolus_agent

def list_folders(root_directory, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Folder', 'Subfolder', 'Depth3Folder', 'FilterType', 'BodyPart', 'ConvolutionKernel', 'CineRate', 'ContrastBolusAgent'])

        for folder in os.listdir(root_directory):
            folder_path = os.path.join(root_directory, folder)

            if os.path.isdir(folder_path):
                for subfolder in os.listdir(folder_path):
                    subfolder_path = os.path.join(folder_path, subfolder)

                    if os.path.isdir(subfolder_path):
                        for depth3_folder in os.listdir(subfolder_path):
                            depth3_path = os.path.join(subfolder_path, depth3_folder)
                            if os.path.isdir(depth3_path):
                                filter_type, body_part, convolution_kernel, cine_rate, contrast_bolus_agent = extract_dicom_info(depth3_path)
                                csv_writer.writerow([folder, subfolder, depth3_folder, filter_type, body_part, convolution_kernel, cine_rate, contrast_bolus_agent])


def generate_texture_config(root_directory, output_file):
    with open(output_file, 'w') as config_file:
        session_number = 0
        for patient_folder in os.listdir(root_directory):
            if os.path.isdir(os.path.join(root_directory, patient_folder)):
                scan_file = None
                roi_file = None

                patient_path = os.path.join(root_directory, patient_folder)
                for dirpath, _, filenames in os.walk(patient_path):
                    for filename in filenames:
                        if filename.endswith("scan.nii.gz"):
                            scan_file = os.path.join(dirpath, filename)
                        elif filename.endswith(".nii"):
                            roi_file = os.path.join(dirpath, filename)

                if scan_file and roi_file:
                    config_file.write(f"LIFEx.texture.Session{session_number}.Img0={scan_file}\n")
                    config_file.write(f"LIFEx.texture.Session{session_number}.Roi0={roi_file}\n")
                    session_number += 1

                    # Add two empty lines
                    config_file.write("\n\n")



root_dir = '/media/adamdiakite/LaCie/Patients_Groupe_BC'
output_file = '/home/adamdiakite/Bureau/gpBf.txt'

generate_texture_config(root_dir, output_file)

# copy_folders_with_subfolder_structure('/media/adamdiakite/LaCie/6-Lille_reformater_trier_contourer_VS', 'Paren','/media/adamdiakite/LaCie/Patients_Groupe_A' )
#
# Example usage:
# root_directory = '/media/adamdiakite/LaCie/6-Lille_reformater_trier_contourer_VS'
# output_csv = '/media/adamdiakite/LaCie/6-Lille_reformater_trier_contourer_VS/folder_list_dicom.csv'
#
# list_folders(root_directory, output_csv)