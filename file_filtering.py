import os
import shutil
import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.widgets import Slider
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

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


def delete_folders_with_characters(folder_path, characters):
    """
    Deletes subfolders in the given folder path and its subdirectories that contain the specified characters in their names.
    :param folder_path: Path to the folder.
    :param characters: String of characters to match in subfolder names.
    """
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for subfolder in dirs:
            if characters in subfolder:
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


def load_img(folder, save_dir):
    """
    Loads and displays CT, PET, and segmentation mask images from the specified folder with a slider for slice selection.
    Plots the resized tumor slices where there is a tumor and saves them in separate CT and PET folders.
    """
    image_folder = os.path.join(folder, "Images")
    ct_folder = os.path.join(image_folder, "CTnii")
    pet_folder = os.path.join(image_folder, "PETnii")
    segmentation_folder = os.path.join(folder, "segmentation")

    # Load CT image
    ct_path = os.path.join(ct_folder, os.listdir(ct_folder)[0])
    ct_img = nib.load(ct_path)
    ct_data = ct_img.get_fdata()
    ct_data = cv2.resize(ct_data, (512, 512), interpolation=cv2.INTER_CUBIC)

    # Load PET image
    pet_path = os.path.join(pet_folder, os.listdir(pet_folder)[0])
    pet_img = nib.load(pet_path)
    pet_data = pet_img.get_fdata()
    pet_data = cv2.resize(pet_data, (512, 512), interpolation=cv2.INTER_CUBIC)

    # Load segmentation mask
    seg_path = os.path.join(segmentation_folder, os.listdir(segmentation_folder)[0])
    seg_img = nib.load(seg_path)
    seg_data = seg_img.get_fdata()
    seg_data = cv2.resize(seg_data, (512, 512), interpolation=cv2.INTER_CUBIC)

    # Rotate images clockwise
    ct_data = np.rot90(ct_data, k=1, axes=(0, 1))
    pet_data = np.rot90(pet_data, k=1, axes=(0, 1))
    seg_data = np.rot90(seg_data, k=1, axes=(0, 1))

    print("CT Shape:", ct_data.shape)
    print("PET Shape:", pet_data.shape)
    print("Segmentation Mask Shape:", seg_data.shape)

    # Create figure and axes for CT, PET, segmentation mask, CT tumor, and PET tumor images
    fig, (ax_ct, ax_pet, ax_seg, ax_ct_tumor, ax_pet_tumor) = plt.subplots(1, 5, figsize=(18, 4))
    fig.suptitle('CT, PET, and Segmentation Mask Images')

    # Display initial slices
    ct_slice = 0
    pet_slice = 0
    seg_slice = 0

    # Display CT image
    ax_ct.imshow(ct_data[:, :, ct_slice], cmap='gray', origin='lower')
    ax_ct.set_title('CT Image')

    # Display PET image
    ax_pet.imshow(pet_data[:, :, pet_slice], cmap='hot', origin='lower', vmin=np.min(pet_data), vmax=np.max(pet_data))
    ax_pet.set_title('PET Image')

    # Display segmentation mask
    ax_seg.imshow(seg_data[:, :, seg_slice], cmap='gray', origin='lower', vmin=np.min(seg_data), vmax=np.max(seg_data))
    ax_seg.set_title('Segmentation Mask')

    # Find tumor coordinates from segmentation mask
    tumor_indices = np.where(seg_data > 0)
    min_y, max_y = np.min(tumor_indices[0]), np.max(tumor_indices[0])
    min_x, max_x = np.min(tumor_indices[1]), np.max(tumor_indices[1])
    tumor_height = max_y - min_y
    tumor_width = max_x - min_x
    tumor_center_y = (max_y + min_y) // 2
    tumor_center_x = (max_x + min_x) // 2

    # Define zoom region around tumor
    zoom_size = 128
    zoom_y_start = max(0, tumor_center_y - zoom_size // 2)
    zoom_y_end = min(ct_data.shape[0], zoom_y_start + zoom_size)
    zoom_x_start = max(0, tumor_center_x - zoom_size // 2)
    zoom_x_end = min(ct_data.shape[1], zoom_x_start + zoom_size)

    # Extract tumor region from CT scan and apply zoom
    ct_tumor_zoomed = ct_data[zoom_y_start:zoom_y_end, zoom_x_start:zoom_x_end, ct_slice]
    ct_tumor_resized = cv2.resize(ct_tumor_zoomed, (zoom_size, zoom_size), interpolation=cv2.INTER_CUBIC)

    # Extract tumor region from PET scan and apply zoom
    pet_tumor_zoomed = pet_data[zoom_y_start:zoom_y_end, zoom_x_start:zoom_x_end, pet_slice]
    pet_tumor_resized = cv2.resize(pet_tumor_zoomed, (zoom_size, zoom_size), interpolation=cv2.INTER_CUBIC)

    # Display CT tumor image
    ax_ct_tumor.imshow(ct_tumor_resized, cmap='gray', origin='lower')
    ax_ct_tumor.set_title('CT Tumor')

    # Display PET tumor image
    ax_pet_tumor.imshow(pet_tumor_resized, cmap='hot', origin='lower', vmin=np.min(seg_data), vmax=np.max(seg_data))
    ax_pet_tumor.set_title('PET Tumor')

    # Adjust spacing and layout
    fig.tight_layout()

    num_slices = pet_data.shape[2]

    # Create slider
    slider_ax = plt.axes([0.1, 0.05, 0.8, 0.03])
    slider = Slider(slider_ax, 'Slice', 0, num_slices - 1, valinit=0, valstep=1)

    # Flag to track if the slider has been closed
    slider_closed = False

    # Update function for slider
    def update(val):
        nonlocal slider_closed
        current_slice = int(slider.val)
        ax_ct.images[0].set_array(ct_data[:, :, current_slice])
        ax_pet.images[0].set_array(pet_data[:, :, current_slice])
        ax_seg.images[0].set_array(seg_data[:, :, current_slice])
        zoom_y_start = max(0, tumor_center_y - zoom_size // 2)
        zoom_y_end = min(ct_data.shape[0], zoom_y_start + zoom_size)
        zoom_x_start = max(0, tumor_center_x - zoom_size // 2)
        zoom_x_end = min(ct_data.shape[1], zoom_x_start + zoom_size)
        ct_tumor_zoomed = ct_data[zoom_y_start:zoom_y_end, zoom_x_start:zoom_x_end, current_slice]
        ct_tumor_resized = cv2.resize(ct_tumor_zoomed, (zoom_size, zoom_size), interpolation=cv2.INTER_LINEAR)
        ax_ct_tumor.images[0].set_array(ct_tumor_resized)
        pet_tumor_zoomed = pet_data[zoom_y_start:zoom_y_end, zoom_x_start:zoom_x_end, current_slice]
        pet_tumor_resized = cv2.resize(pet_tumor_zoomed, (zoom_size, zoom_size), interpolation=cv2.INTER_LINEAR)
        ax_pet_tumor.images[0].set_array(pet_tumor_resized)
        fig.canvas.draw_idle()

    # Connect slider update function to slider
    slider.on_changed(update)

    # Show the plots
    plt.show()

    # Check if the slider has been closed
    if not slider_closed:
        # Find tumor slices based on segmentation mask
        tumor_slices = np.unique(np.where(seg_data > 1)[2])

        # Create CT and PET tumor directories
        ct_save_dir = os.path.join(save_dir, "CT")
        pet_save_dir = os.path.join(save_dir, "PET")
        os.makedirs(ct_save_dir, exist_ok=True)
        os.makedirs(pet_save_dir, exist_ok=True)

        # Save the corresponding CT tumor images for tumor slices
        for slice_index in tumor_slices:
            ct_tumor_slice = ct_data[:, :, slice_index]
            ct_tumor_zoomed = ct_tumor_slice[zoom_y_start:zoom_y_end, zoom_x_start:zoom_x_end]
            ct_tumor_resized = cv2.resize(ct_tumor_zoomed, (zoom_size, zoom_size), interpolation=cv2.INTER_CUBIC)
            save_path = os.path.join(ct_save_dir, "Slice_{}.png".format(slice_index))
            plt.imsave(save_path, ct_tumor_resized, cmap='gray', origin='lower')

        # Save the corresponding PET tumor images for tumor slices
        for slice_index in tumor_slices:
            pet_tumor_slice = pet_data[:, :, slice_index]
            pet_tumor_zoomed = pet_tumor_slice[zoom_y_start:zoom_y_end, zoom_x_start:zoom_x_end]
            pet_tumor_resized = cv2.resize(pet_tumor_zoomed, (zoom_size, zoom_size), interpolation=cv2.INTER_CUBIC)
            save_path = os.path.join(pet_save_dir, "Slice_{}.png".format(slice_index))
            plt.imsave(save_path, pet_tumor_resized, cmap='hot', origin='lower', vmin=np.min(seg_data),
                       vmax=np.max(seg_data))

        # Print the indices of the tumor slices
        print("Tumor slice indices:", tumor_slices)


# Example usage:
folder_path = "/home/adamdiakite/Documents/2-21-0004"
images_curie = "/home/adamdiakite/Documents/lungegfr-master/Sampledata/Images_Curie"
folder = "/media/lito/LaCie/CT-TEP_Data/"

load_img(folder_path, images_curie)
