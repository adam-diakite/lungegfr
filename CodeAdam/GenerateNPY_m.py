import os
import numpy as np
from PIL import Image
import csv
import nibabel as nib
from tqdm import tqdm
from skimage.transform import resize
def normalization_img(image):
    mean_value = np.mean(image)
    std_value = np.std(image)

    if std_value == 0:
        # Handle the case when the standard deviation is zero (constant image)
        return np.zeros_like(image)

    normalized_image = (image - mean_value) / std_value

    # Handle NaN or infinite values in the normalized image
    normalized_image[np.isnan(normalized_image)] = 0
    normalized_image[np.isinf(normalized_image)] = 0

    return normalized_image


def calculate_tumor_areas(seg_path):
    # Load the segmentation NIfTI file
    seg_img = nib.load(seg_path)
    seg_data = seg_img.get_fdata()

    # Check if the segmentation data is binary
    is_binary = np.all(np.logical_or(seg_data == 0, seg_data == 1))

    # Calculate the range of values in the segmentation data
    value_range = (np.min(seg_data), np.max(seg_data))

    tumor_areas = []
    for slice_idx in range(seg_data.shape[2]):
        slice_data = seg_data[:, :, slice_idx]

        # Resize the slice_data to a height and width of 500
        slice_data_resized = resize(slice_data, (500, 500), preserve_range=True, anti_aliasing=False)

        # Binarize the slice data if it's not already binary
        if not is_binary:
            slice_data_resized = np.where(slice_data_resized > 0.5, 1, 0)

        tumor_area = np.sum(slice_data_resized)

        # Include the tumor area in the list only if it's greater than 0
        if tumor_area > 0:
            tumor_areas.append(tumor_area)

    print("Shape of segmentation data:", seg_data.shape)
    print("total number of slices: ", len(tumor_areas))

    print("Range of values in the segmentation data:", value_range)
    print("Is the segmentation data binary?", is_binary)

    return tumor_areas




def process_patient(root, ct_png_folder, pet_png_folder, seg_nii_folder, patient_output_folder):
    patient_name = os.path.basename(root)

    ct_files = sorted(os.listdir(ct_png_folder))
    pet_files = sorted(os.listdir(pet_png_folder))
    seg_files = sorted(os.listdir(seg_nii_folder))

    if len(ct_files) != len(pet_files):
        print(f"Patient {patient_name}: Mismatch in the number of files for CT and PET. Skipping.")
        return

    num_ct_slices = len(ct_files)
    num_pet_slices = len(pet_files)

    print(f"Patient: {patient_name}")
    print(f"Number of CT slices: {num_ct_slices}")
    print(f"Number of PET slices: {num_pet_slices}")

    # Use the minimum number of slices between CT and PET for the segmentation file
    num_seg_slices = min(num_ct_slices, num_pet_slices)

    print(f"Number of Segmentation slices: {num_seg_slices}")

    num_files = num_seg_slices
    cts = np.empty((num_files, 64, 64), dtype=np.float64)
    pets = np.empty((num_files, 64, 64), dtype=np.float64)
    fuses = np.empty((num_files, 64, 64), dtype=np.float64)
    tumor_areas_per_slice = []

    for i, (ct_file, pet_file, seg_file) in enumerate(tqdm(zip(ct_files[:num_files], pet_files[:num_files], seg_files[:num_files]), total=num_files, desc=f"Processing {patient_name}")):
        ct_path = os.path.join(ct_png_folder, ct_file)
        pet_path = os.path.join(pet_png_folder, pet_file)
        seg_path = os.path.join(seg_nii_folder, seg_file)

        ct_image = np.array(Image.open(ct_path), dtype=np.float64)
        pet_image = np.array(Image.open(pet_path), dtype=np.float64)

        if len(ct_image.shape) > 2:
            ct_image = ct_image[:, :, 0]
        if len(pet_image.shape) > 2:
            pet_image = pet_image[:, :, 0]

        ct_image = normalization_img(np.array(Image.fromarray(ct_image).resize((64, 64), Image.NEAREST)))
        pet_image = normalization_img(np.array(Image.fromarray(pet_image).resize((64, 64), Image.NEAREST)))

        fuse_image = pet_image + ct_image
        fuse_image = normalization_img(fuse_image)

        cts[i] = ct_image
        pets[i] = pet_image
        fuses[i] = fuse_image

        # Calculate tumor area using the calculate_tumor_areas function
        tumor_area_slice = calculate_tumor_areas(seg_path)
        tumor_areas_per_slice.append(tumor_area_slice)

    print(f"Tumor areas per slice for patient {patient_name}: {tumor_areas_per_slice}")

    # Save tumor areas to a CSV file
    csv_file_path = os.path.join(patient_output_folder, 'tumor_areas.csv')
    with open(csv_file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Slice number', 'Tumor Area'])
        for i, areas in enumerate(tumor_areas_per_slice):
            for area in areas:
                csvwriter.writerow([area + 1, 1])  # Only store the index where the area is different from 0

    # Create the 'label.npy' file with random 0s and 1s
    label = np.random.randint(2, size=num_files)
    np.save(os.path.join(patient_output_folder, 'ct.npy'), cts)
    np.save(os.path.join(patient_output_folder, 'pet.npy'), pets)
    np.save(os.path.join(patient_output_folder, 'fuse.npy'), fuses)
    np.save(os.path.join(patient_output_folder, 'label.npy'), label)

    print(f"Arrays, tumor areas per slice, and label for patient {patient_name} saved successfully.")

def process_patient_area(root, seg_nii_folder, patient_output_folder):
    patient_name = os.path.basename(root)

    seg_files = sorted(os.listdir(seg_nii_folder))

    num_files = len(seg_files)
    tumor_areas_per_slice = []

    for i, seg_file in enumerate(tqdm(seg_files, total=num_files, desc=f"Processing {patient_name}")):
        seg_path = os.path.join(seg_nii_folder, seg_file)

        # Calculate tumor area using the calculate_tumor_areas function
        tumor_area_slice = calculate_tumor_areas(seg_path)
        tumor_areas_per_slice.append(tumor_area_slice)

    print(f"Tumor areas per slice for patient {patient_name}: {tumor_areas_per_slice}")

    # Save tumor areas to a CSV file
    csv_file_path = os.path.join(patient_output_folder, 'tumor_areas.csv')
    with open(csv_file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Slice Number', 'Tumor Area'])
        for i, areas in enumerate(tumor_areas_per_slice):
            for area in areas:
                csvwriter.writerow([i + 1, area])

    print(f"Tumor areas CSV for patient {patient_name} saved successfully.")

def process_png_folders(root_folder, output_folder):
    """
    Processes the "CT_PNG" and "PET_PNG" folders for each subdirectory in the root folder.
    Applies normalization and saves the arrays for each pair of folders.

    :param root_folder: The root directory to start searching.
    :param output_folder: The folder to save the numpy arrays.
    """
    for root, dirs, _ in os.walk(root_folder):
        if "Images" in dirs:
            ct_png_folder = os.path.join(root, "Images", "CT_PNG")
            pet_png_folder = os.path.join(root, "Images", "PET_PNG")
            seg_nii_folder = os.path.join(root, "Segmentations")

            if os.path.exists(ct_png_folder) and os.path.exists(pet_png_folder) and os.path.exists(seg_nii_folder):
                patient_name = os.path.basename(root)
                patient_output_folder = os.path.join(output_folder, patient_name)
                os.makedirs(patient_output_folder, exist_ok=True)

                process_patient(root, ct_png_folder, pet_png_folder, seg_nii_folder, patient_output_folder)
            else:
                print("Both CT_PNG, PET_PNG, and segmentation folders are not present in the current folder.")
                continue

def process_png_folders_areas(root_folder, output_folder):
    """
    Processes the "CT_PNG" and "PET_PNG" folders for each subdirectory in the root folder.
    Applies normalization and saves the arrays for each pair of folders.

    :param root_folder: The root directory to start searching.
    :param output_folder: The folder to save the numpy arrays.
    """
    for root, dirs, _ in os.walk(root_folder):
        if "Images" in dirs:
            seg_nii_folder = os.path.join(root, "segmentation")

            if os.path.exists(seg_nii_folder):
                patient_name = os.path.basename(root)
                patient_output_folder = os.path.join(output_folder, patient_name)
                os.makedirs(patient_output_folder, exist_ok=True)

                process_patient_area(root, seg_nii_folder, patient_output_folder)
            else:
                print("Segmentation folder is not present in the current folder.")
                continue


if __name__ == "__main__":
    folder_ICI = "/media/adamdiakite/LaCie/CT-TEP_TKI"
    dest_ICI = "/media/adamdiakite/LaCie/CT-TEP_TKI/NPY"
    process_png_folders_areas(folder_ICI, dest_ICI)

