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



def nifti_processing(root_folder):
    """
    Processes the NIfTI files in each patient's folder at the specified location.
    Returns the necessary information for displaying every slice of CT, PET, and segmentation NIfTI files.
    """
    patients_info = []
    patients_with_error = []  # To store patient folders that encountered EOFError

    for root, dirs, _ in os.walk(root_folder):
        if "Images" in dirs and "segmentation" in dirs:
            print(f"Processing directory: {root}")
            ct_folder = os.path.join(root, "Images", "CTnii")
            pet_folder = os.path.join(root, "Images", "PETnii")
            segmentation_folder = os.path.join(root, "segmentation")

            ct_files = os.listdir(ct_folder)
            pet_files = os.listdir(pet_folder)
            segmentation_files = [f for f in os.listdir(segmentation_folder) if f.endswith(".nii.gz")]

            if len(ct_files) == 1 and len(pet_files) == 1 and len(segmentation_files) == 1:
                ct_nii_file = os.path.join(ct_folder, ct_files[0])
                pet_nii_file = os.path.join(pet_folder, pet_files[0])
                segmentation_nii_file = os.path.join(segmentation_folder, segmentation_files[0])

                try:
                    nifti_ct = nib.load(ct_nii_file)
                    nifti_pet = nib.load(pet_nii_file)
                    nifti_segmentation = nib.load(segmentation_nii_file)

                    # Resample CT image to match PET and segmentation resolution
                    nifti_ct_resampled = resample_img(nifti_ct, target_affine=nifti_pet.affine, target_shape=nifti_pet.shape)

                    # Binarize the segmentation file
                    seg_data = nifti_segmentation.get_fdata()
                    seg_data_binary = (seg_data >= 0.5).astype(np.uint8)

                    # Find tumor coordinates from segmentation mask
                    tumor_indices = np.where(seg_data_binary == 1)
                    min_y, max_y = np.min(tumor_indices[0]), np.max(tumor_indices[0])
                    min_x, max_x = np.min(tumor_indices[1]), np.max(tumor_indices[1])
                    min_z, max_z = np.min(tumor_indices[2]), np.max(tumor_indices[2])

                    # Ensure the bounding box size is 128x128
                    center_y, center_x = (min_y + max_y) // 2, (min_x + max_x) // 2
                    half_size = 32  # half of the desired size (128/2 = 64)

                    # Create a new bounding box that is centered around the tumor and has a size of 128x128
                    new_min_y = center_y - half_size
                    new_max_y = center_y + half_size
                    new_min_x = center_x - half_size
                    new_max_x = center_x + half_size

                    # Extract tumor region from CT and PET data based on the new bounding box
                    ct_tumor = nifti_ct_resampled.slicer[new_min_y:new_max_y, new_min_x:new_max_x, min_z:max_z].get_fdata()
                    pet_tumor = nifti_pet.slicer[new_min_y:new_max_y, new_min_x:new_max_x, min_z:max_z].get_fdata()

                    # Ensure the tumor images have a fixed size of 128x128
                    ct_tumor_resized = resize_image(ct_tumor, target_shape=(128, 128, ct_tumor.shape[-1]))
                    pet_tumor_resized = resize_image(pet_tumor, target_shape=(128, 128, pet_tumor.shape[-1]))

                    patient_info = {
                        "patient": root,
                        "ct_nii": nifti_ct_resampled,
                        "pet_nii": nifti_pet,
                        "segmentation_nii": nifti_segmentation,
                        "ct_tumor": ct_tumor_resized,
                        "pet_tumor": pet_tumor_resized,
                        "seg_tumor": seg_data[new_min_y:new_max_y, new_min_x:new_max_x, min_z:max_z],
                        "tumor_bbox": (new_min_y, new_max_y, new_min_x, new_max_x, min_z, max_z),
                        "images_folder": os.path.join(root, "Images"),
                    }
                    patients_info.append(patient_info)
                    print(f"Patient {root} processed")
                except EOFError:
                    patients_with_error.append(root)
                    print(f"EOFError: Skipping patient {root} due to incomplete NIfTI files.")
                    continue

    if patients_with_error:
        print("Patients with EOFError:")
        for patient_folder in patients_with_error:
            print(patient_folder)

    return patients_info


def resize_image(image, target_shape):
    """
    Resizes the given image to the target shape.
    :param image: Image to resize.
    :param target_shape: Tuple specifying the target shape.
    :return: Resized image.
    """
    resized_image = resize(image, target_shape, mode='constant', anti_aliasing=True)

    return resized_image

def normalize_image(image):
    # Normalize the image to 0-255
    image_min = np.min(image)
    image_max = np.max(image)
    image_range = image_max - image_min
    image_normalized = ((image - image_min) / image_range) * 255
    return image_normalized.astype(np.uint8)

def save_image_as_png(image, output_folder, filename):
    image_normalized = normalize_image(image)

    # Rotate clockwise three times to get 90 degrees clockwise rotation
    image_normalized_rotated = np.rot90(image_normalized, k=3)

    # Flip vertically to match medical image convention (origin='lower')
    image_normalized_flipped = np.flipud(image_normalized_rotated)

    img = Image.fromarray(image_normalized_flipped)
    img.save(os.path.join(output_folder, filename))

def save_ct_pet_images(patients_info):
    for patient_info in patients_info:
        nifti_ct = patient_info["ct_nii"]
        nifti_pet = patient_info["pet_nii"]
        tumor_bbox = patient_info["tumor_bbox"]
        images_folder = patient_info["images_folder"]

        ct_data = nifti_ct.get_fdata()
        pet_data = nifti_pet.get_fdata()

        min_y, max_y, min_x, max_x, min_z, max_z = tumor_bbox

        # Extract tumor region from CT and PET scan based on bounding box
        ct_tumor = ct_data[min_y:max_y, min_x:max_x, min_z:max_z]
        pet_tumor = pet_data[min_y:max_y, min_x:max_x, min_z:max_z]

        # Create output folders for CT and PET tumors
        ct_tumor_folder = os.path.join(images_folder, "CT_PNG")
        pet_tumor_folder = os.path.join(images_folder, "PET_PNG")
        os.makedirs(ct_tumor_folder, exist_ok=True)
        os.makedirs(pet_tumor_folder, exist_ok=True)

        # Save CT tumor slices
        for i in range(ct_tumor.shape[2]):
            ct_tumor_filename = f"ct_tumor_slice_{i:03d}.png"
            save_image_as_png(ct_tumor[:, :, i], ct_tumor_folder, ct_tumor_filename)

        # Save PET tumor slices
        for i in range(pet_tumor.shape[2]):
            pet_tumor_filename = f"pet_tumor_slice_{i:03d}.png"
            save_image_as_png(pet_tumor[:, :, i], pet_tumor_folder, pet_tumor_filename)


def process_patient_folder(patient_folder):
    patients_info = nifti_processing(patient_folder)
    save_ct_pet_images(patients_info)

def display_nifti_slices(ct_nii_file, pet_nii_file, segmentation_nii_file):
    # Load NIfTI files
    ct_nii = nib.load(ct_nii_file)
    pet_nii = nib.load(pet_nii_file)
    segmentation_nii = nib.load(segmentation_nii_file)

    # Get the data from NIfTI files
    ct_data = ct_nii.get_fdata()
    pet_data = pet_nii.get_fdata()
    segmentation_data = segmentation_nii.get_fdata()

    # Rotate images clockwise three times
    ct_data = ct_data.transpose(1, 0, 2)[:, ::-1, :][::-1, :, :]
    pet_data = pet_data.transpose(1, 0, 2)[:, ::-1, :][::-1, :, :]
    segmentation_data = segmentation_data.transpose(1, 0, 2)[:, ::-1, :][::-1, :, :]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Display CT slice
    ct_slice = axes[0].imshow(ct_data[..., 0], cmap='gray')
    axes[0].set_title('CT Slice')
    axes[0].axis('off')

    # Display PET slice
    pet_slice = axes[1].imshow(pet_data[..., 0], cmap='hot')
    axes[1].set_title('PET Slice')
    axes[1].axis('off')

    # Display Segmentation slice with proper colormap
    seg_slice = axes[2].imshow(segmentation_data[..., 0], cmap='jet', alpha=0.7, vmin=0, vmax=1)
    axes[2].set_title('Segmentation Slice')
    axes[2].axis('off')

    plt.tight_layout()

    # Function to update the displayed slices when slider value changes
    def update_slices(val):
        index = int(val)
        ct_slice.set_data(ct_data[..., index])
        pet_slice.set_data(pet_data[..., index])
        seg_slice.set_data(segmentation_data[..., index])
        fig.canvas.draw_idle()

    # Create a slider for selecting the slice index
    slice_slider_ax = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slice_slider = Slider(slice_slider_ax, 'Slice', 0, ct_data.shape[-1] - 1, valinit=0, valstep=1)

    # Connect the slider to the update_slices function
    slice_slider.on_changed(update_slices)

    plt.show()

ct_nii_file = '/media/adamdiakite/LaCie/CT-TEP_Data/2-21-0011/Images/CTnii/3_body-ldct.nii.gz'
pet_nii_file = '/media/adamdiakite/LaCie/CT-TEP_Data/2-21-0011/Images/PETnii/2-21-0011_pet_float32_SUVmax.nii.gz'
segmentation_nii_file = '/media/adamdiakite/LaCie/CT-TEP_Data/2-21-0011/segmentation/PRIMITIF_PULM_Abs_thres4.0to999.0.uint16.nii.gz'

display_nifti_slices(ct_nii_file, pet_nii_file, segmentation_nii_file)

# if __name__ == "__main__":
#     root_folder = "/media/adamdiakite/LaCie/CT-TEP_Data"
#     for folder_name in os.listdir(root_folder):
#         patient_folder = os.path.join(root_folder, folder_name)
#         if os.path.isdir(patient_folder):
#             print(f"Processing patient folder: {patient_folder}")
#             process_patient_folder(patient_folder)

