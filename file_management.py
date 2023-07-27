import os
import shutil
import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import nibabel.processing
import numpy as np
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import rotate
import matplotlib.patches as patches
from nilearn.image import resample_to_img
from scipy.ndimage import zoom
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from nilearn.image import resample_img
from skimage.transform import resize
from matplotlib.widgets import Slider

directory_path = "/media/adamdiakite/LaCie/batch_TEP_PP_020522"  # Replace with your directory path

source_directory = "/media/adamdiakite/LaCie/batch_TEP_PP_020522"  # Replace with the path to the source directory
destination_directory = "/media/adamdiakite/LaCie/CT-TEP_Data"  # Replace with the path to the destination directory

data = "/media/adamdiakite/LaCie/CT-TEP_Data"


def list_patient_folders(directory):
    """
    Lists all patient folder names within a directory.
    :param directory: The directory path.
    :return: A string with patient folder names separated by a newline character.
    """
    patient_folders = []

    # List all items in the directory
    items = os.listdir(directory)

    # Iterate over the items and check if they are directories
    for item in items:
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            patient_folders.append(item)

    # Join the patient folder names with newline character
    patient_folders_string = '\n'.join(patient_folders)

    return patient_folders_string


def create_patients_without_subfolders(source_directory, destination_directory):
    """
    Creates the same patient folders without their subfolders at the specified destination directory.
    :param source_directory: The source directory containing the patient folders.
    :param destination_directory: The destination directory where the new patient folders will be created.
    """
    # List all patient folders in the source directory
    patient_folders = [folder for folder in os.listdir(source_directory) if
                       os.path.isdir(os.path.join(source_directory, folder))]

    # Create patient folders at the destination directory
    for folder in patient_folders:
        destination_path = os.path.join(destination_directory, folder)
        os.makedirs(destination_path)

    print("Patient folders created without subfolders successfully.")


def copy_image_folders(source_directory, destination_directory):
    """
    Copies the "Images" folder from the source directory to the destination directory
    while maintaining the same patient folder structure.
    :param source_directory: The source directory containing the patient folders.
    :param destination_directory: The destination directory where the "Images" folders will be copied.
    """
    # List all patient folders in the source directory
    patient_folders = [folder for folder in os.listdir(source_directory) if
                       os.path.isdir(os.path.join(source_directory, folder))]

    # Copy "Images" folder for each patient to the destination directory
    for folder in patient_folders:
        source_images_folder = os.path.join(source_directory, folder, 'Images')
        destination_images_folder = os.path.join(destination_directory, folder, 'Images')

        # Skip if "Images" folder already exists in the destination
        if os.path.exists(destination_images_folder):
            print(f"Skipping {folder}: 'Images' folder already exists in the destination.")
            continue

        # Skip if "Images" folder does not exist in the source
        if not os.path.exists(source_images_folder):
            print(f"Skipping {folder}: 'Images' folder does not exist in the source.")
            continue

        # Copy the "Images" folder to the destination
        shutil.copytree(source_images_folder, destination_images_folder)
        print(f"Successfully copied 'Images' folder for {folder}.")

    print("Image folders copied successfully.")


def copy_segmentation_file(source_directory, destination_directory):
    """
    Copies the "PRIMITIF" file from the "SUV4" folder in the source directory to the "segmentation" folder
    in the destination directory for each patient.
    :param source_directory: The source directory containing the patient folders.
    :param destination_directory: The destination directory where the "segmentation" folders will be created
                                  and the "PRIMITIF" file will be copied.
    """
    # List all patient folders in the source directory
    patient_folders = [folder for folder in os.listdir(source_directory) if
                       os.path.isdir(os.path.join(source_directory, folder))]

    # Copy "PRIMITIF" file for each patient to the destination directory
    for folder in patient_folders:
        source_segmentation_folder = os.path.join(source_directory, folder, 'Segmentations')

        if not os.path.exists(source_segmentation_folder):
            print(f"Skipping {folder}: No 'Segmentations' folder found in the source directory.")
            continue

        # Find the folder containing "SUV4" in its name
        suv4_folder = None
        for subfolder in os.listdir(source_segmentation_folder):
            if 'SUV4' in subfolder:
                suv4_folder = subfolder
                break

        if suv4_folder is None:
            print(f"Skipping {folder}: No 'SUV4' folder found in the segmentation directory.")
            continue

        # Find the "PRIMITIF" file in the "SUV4" folder
        suv4_folder_path = os.path.join(source_segmentation_folder, suv4_folder)
        primitif_file = None
        for file in os.listdir(suv4_folder_path):
            if 'PRIMITIF' in file:
                primitif_file = file
                break

        if primitif_file is None:
            print(f"Skipping {folder}: No 'PRIMITIF' file found in the 'SUV4' folder.")
            continue

        destination_segmentation_folder = os.path.join(destination_directory, folder, 'segmentation')

        if os.path.exists(destination_segmentation_folder):
            print(f"Skipping {folder}: Destination 'segmentation' folder already exists.")
            continue

        # Create the destination "segmentation" folder
        os.makedirs(destination_segmentation_folder)

        # Copy the "PRIMITIF" file to the destination segmentation folder
        source_file_path = os.path.join(suv4_folder_path, primitif_file)
        destination_file_path = os.path.join(destination_segmentation_folder, primitif_file)
        shutil.copy2(source_file_path, destination_file_path)
        print(f"Successfully copied 'PRIMITIF' file for {folder}.")

    print("Segmentation files copied successfully.")


def delete_patient_folders_without_segmentation(destination_directory):
    """
    Deletes all patient folders in the destination directory that don't contain a 'segmentation' folder.
    :param destination_directory: The destination directory where the patient folders are located.
    """
    # List all patient folders in the destination directory
    patient_folders = [folder for folder in os.listdir(destination_directory) if
                       os.path.isdir(os.path.join(destination_directory, folder))]

    # Check each patient folder for the existence of a 'segmentation' folder
    for folder in patient_folders:
        segmentation_folder = os.path.join(destination_directory, folder, 'segmentation')

        if not os.path.exists(segmentation_folder):
            # Delete the patient folder if 'segmentation' folder doesn't exist
            patient_folder_path = os.path.join(destination_directory, folder)
            shutil.rmtree(patient_folder_path)
            print(f"Deleted {folder} as it doesn't contain a 'segmentation' folder.")

    print("Patient folders without segmentation deleted successfully.")


def copy_patient_image_folders(root_folder):
    """
    Copies all subfolders in the 'Images' folder of each patient folder to the same location with '_1' added to the name.
    :param root_folder: The root folder containing the patient folders.
    """
    patient_folders = [folder for folder in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, folder))]

    for patient_folder in patient_folders:
        images_folder = os.path.join(root_folder, patient_folder, 'Images')

        if not os.path.exists(images_folder):
            continue

        # Copy all subfolders in the Images folder
        for folder in os.listdir(images_folder):
            folder_path = os.path.join(images_folder, folder)

            # Copy the subfolder to the same location with '_1' added to the name
            if os.path.isdir(folder_path):
                new_folder_path = os.path.join(images_folder, folder + '_1')
                shutil.copytree(folder_path, new_folder_path)

    print("Image folders copied successfully.")


def delete_files_containing_string(root_folder, target_string):
    for root, dirs, files in os.walk(root_folder):
        for file_name in files:
            if target_string in file_name:
                file_path = os.path.join(root, file_name)
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")


def delete_files_containing_string(root_folder, target_string):
    for root, dirs, files in os.walk(root_folder):
        for file_name in files:
            if target_string in file_name:
                file_path = os.path.join(root, file_name)
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")


def delete_folders_with_characters(folder_path, characters):
    """
    Deletes subfolders in the given folder path and its subdirectories that have the same name as the specified characters.
    :param folder_path: Path to the folder.
    :param characters: String of characters to match in subfolder names.
    """
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for subfolder in dirs:
            if subfolder == characters:
                subfolder_path = os.path.join(root, subfolder)
                # Delete the subfolder and its contents
                for sub_root, sub_dirs, sub_files in os.walk(subfolder_path, topdown=False):
                    for file_name in sub_files:
                        file_path = os.path.join(sub_root, file_name)
                        os.remove(file_path)
                    for dir_name in sub_dirs:
                        dir_path = os.path.join(sub_root, dir_name)
                        os.rmdir(dir_path)
                os.rmdir(subfolder_path)

    print("Deletion of folders completed successfully.")





