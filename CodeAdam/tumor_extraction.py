import os
import shutil
import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import nibabel.processing
import numpy as np
from PIL import Image
from nilearn.image import resample_img
from skimage.transform import resize
from matplotlib.widgets import Slider
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

def nifti_processing(root_folder):
    patients_info = []
    patients_with_error = []

    for root, dirs, _ in os.walk(root_folder):
        if "Images" in dirs and "Segmentations" in dirs:
            print(f"Processing directory: {root}")
            ct_folder = os.path.join(root, "Images", "CTnii")
            pet_folder = os.path.join(root, "Images", "PETnii")
            segmentation_folder = os.path.join(root, "Segmentations")

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

                    # Calculate tumor area
                    tumor_area = np.count_nonzero(seg_data_binary)

                    # Find tumor coordinates from segmentation mask
                    tumor_indices = np.where(seg_data_binary == 1)
                    min_y, max_y = np.min(tumor_indices[0]), np.max(tumor_indices[0])
                    min_x, max_x = np.min(tumor_indices[1]), np.max(tumor_indices[1])
                    min_z, max_z = np.min(tumor_indices[2]), np.max(tumor_indices[2])

                    # 64*64 bounding box
                    center_y, center_x = (min_y + max_y) // 2, (min_x + max_x) // 2
                    half_size = 32  # half of the desired size (64/2 = 32)

                    # Create a new bounding box that is centered around the tumor and has a size of 64*64
                    new_min_y = center_y - half_size
                    new_max_y = center_y + half_size
                    new_min_x = center_x - half_size
                    new_max_x = center_x + half_size

                    # Extract tumor region from CT and PET data based on the new bounding box
                    ct_tumor = nifti_ct_resampled.slicer[new_min_y:new_max_y, new_min_x:new_max_x, min_z:max_z].get_fdata()
                    pet_tumor = nifti_pet.slicer[new_min_y:new_max_y, new_min_x:new_max_x, min_z:max_z].get_fdata()

                    # Ensure the tumor images have a fixed size of 128x128
                    ct_tumor_resized = resize_image(ct_tumor, target_shape=(64, 64, ct_tumor.shape[-1]))
                    pet_tumor_resized = resize_image(pet_tumor, target_shape=(64, 64, pet_tumor.shape[-1]))

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
                        "tumor_area": tumor_area
                    }
                    patients_info.append(patient_info)
                except (EOFError, FileNotFoundError, MemoryError):
                    patients_with_error.append(root)
                    print(f"Error: Skipping patient {root} due to missing or incomplete NIfTI files or memory issues.")
                    continue

    if patients_with_error:
        print("Patients with Errors:")
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

def normalization_img(img):
    """Apply normalization to the image."""
    img = img / 255.0
    return img

def save_image_as_png(image, output_folder, filename):
    image_normalized = normalize_image(image)

    # Rotate clockwise three times to get 90 degrees clockwise rotation
    image_normalized_rotated = np.rot90(image_normalized, k=3)

    # Flip vertically to match medical image convention (origin='lower')
    image_normalized_flipped = np.flipud(image_normalized_rotated)

    img = Image.fromarray(image_normalized_flipped)
    img.save(os.path.join(output_folder, filename))

def save_ct_pet_images(patient_info):
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

    def save_slice(image, output_folder, filename):
        image_normalized = normalize_image(image)

        # Rotate clockwise three times to get 90 degrees clockwise rotation
        image_normalized_rotated = np.rot90(image_normalized, k=3)

        # Flip vertically to match medical image convention (origin='lower')
        image_normalized_flipped = np.flipud(image_normalized_rotated)

        img = Image.fromarray(image_normalized_flipped)
        img.save(os.path.join(output_folder, filename))
        print(f"Saved slice {filename} in {output_folder}")

    n_slices = ct_tumor.shape[2] + pet_tumor.shape[2]

    def save_images_with_progress():
        pbar = tqdm(total=n_slices, desc="Saving PNG Images", unit="slice")
        for i in range(ct_tumor.shape[2]):
            ct_tumor_filename = f"ct_tumor_slice_{i:03d}.png"
            save_slice(ct_tumor[:, :, i], ct_tumor_folder, ct_tumor_filename)
            pbar.update(1)

        for i in range(pet_tumor.shape[2]):
            pet_tumor_filename = f"pet_tumor_slice_{i:03d}.png"
            save_slice(pet_tumor[:, :, i], pet_tumor_folder, pet_tumor_filename)
            pbar.update(1)
        pbar.close()

    save_images_with_progress()

    logging.info(f"All images saved for patient {patient_info['patient']}.")





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


def process_patient_folder(patient_folder):
    images_folder = os.path.join(patient_folder, "Images")
    ct_png_folder = os.path.join(images_folder, "CT_PNG")
    pet_png_folder = os.path.join(images_folder, "PET_PNG")

    # Check if the CT_PNG and PET_PNG folders already exist, if yes, skip the patient
    if os.path.exists(ct_png_folder) and os.path.exists(pet_png_folder):
        logging.info(f"Patient {patient_folder} already has CT_PNG and PET_PNG folders. Skipping.")
        return

    patients_info = nifti_processing(patient_folder)
    num_patients = len(patients_info)
    for i, patient_info in enumerate(patients_info, 1):
        logging.info(f"Processing patient {i}/{num_patients}")
        save_ct_pet_images(patient_info)
        logging.info(f"All images saved for patient {patient_info['patient']}.")

if __name__ == "__main__":
    root_folder = "/media/lito/LaCie/tempp/"

    num_patients = len(os.listdir(root_folder))
    for i, folder_name in enumerate(os.listdir(root_folder), 1):
        patient_folder = os.path.join(root_folder, folder_name)
        if os.path.isdir(patient_folder):
            logging.info(f"Processing patient {i}/{num_patients}")
            process_patient_folder(patient_folder)

    logging.info("All patients processed.")

#
# ct_nii_file = "/media/lito/LaCie/CT-TEP_TKI/2-21-0004/Images/CTnii/2_body-low_dose_ct.nii.gz"
# pet_nii_file = "/media/lito/LaCie/CT-TEP_TKI/2-21-0004/Images/PETnii/2-21-0004_pet_float32_SUVbw.nii.gz"
# segmentation_nii_file = "/media/lito/LaCie/CT-TEP_TKI/2-21-0004/segmentation/PRIMITIF_PULM_Abs_thres4.0to999.0.uint16.nii.gz"
# display_nifti_slices(ct_nii_file, pet_nii_file, segmentation_nii_file)