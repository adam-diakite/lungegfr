import os
import pydicom
import numpy as np
import nibabel as nib
import dicom2nifti
import shutil
from scipy.ndimage import zoom, rotate

# dicom_series_folder = '/media/lito/LaCie/CT-TEP_Data/2-21-0005/Images/CT/'
# nifti_output_file = '/media/lito/LaCie/CT-TEP_Data/2-21-0005/Images/'
# dicom2nifti.convert_directory(dicom_series_folder, nifti_output_file)
#
# nifti_ct = "/media/lito/LaCie/CT-TEP_Data/2-21-0005/Images/ct_nii_ex.nii.gz"
# nifti_pet = "/media/lito/LaCie/CT-TEP_Data/2-21-0005/Images/PETnii/2-21-0005_pet_float32_SUVbw.nii.gz"
#
#
# # Load NIfTI files
# nifti1 = nib.load(nifti_ct)
# nifti2 = nib.load(nifti_pet)
#
# # Get voxel size for each NIfTI file
# voxel_size1 = nifti1.header.get_zooms()
# voxel_size2 = nifti2.header.get_zooms()
#
# # Change voxel size of voxel_size1 to match voxel_size2
# voxel_size1 = voxel_size2
#
# # Print voxel size
# print(f'Voxel size of {nifti_ct}: {voxel_size1}')
# print(f'Voxel size of {nifti_pet}: {voxel_size2}')
#


def convert(root_folder):
    for root, _, _ in os.walk(root_folder):
        ct_folder = os.path.join(root, "Images", "CT")
        if os.path.exists(ct_folder):
            nifti_output_file = os.path.join(root, "Images", "CTnii")
            if not os.path.exists(nifti_output_file):
                os.makedirs(nifti_output_file)

            # Convert DICOM series to NIfTI in the "CTnii" folder
            dicom2nifti.convert_directory(ct_folder, nifti_output_file)
            print(f"File {nifti_output_file} converted")


def resample(root_folder):
    for root, dirs, _ in os.walk(root_folder):
        if "Images" in dirs:
            ct_folder = os.path.join(root, "Images", "CTnii")
            pet_folder = os.path.join(root, "Images", "PETnii")
            segmentation_folder = os.path.join(root, "segmentation")

            # Initialize NIfTI file paths
            ct_nii_file = None
            pet_nii_file = None
            segmentation_nii_file = None

            # Get the CT NIfTI file
            ct_files = os.listdir(ct_folder)
            if len(ct_files) == 1:
                ct_nii_file = os.path.join(ct_folder, ct_files[0])

            # Get the PET NIfTI file
            pet_files = os.listdir(pet_folder)
            if len(pet_files) == 1:
                pet_nii_file = os.path.join(pet_folder, pet_files[0])

            # Get the Segmentation NIfTI file
            segmentation_files = os.listdir(segmentation_folder)
            segmentation_files = [f for f in segmentation_files if f.endswith(".nii.gz")]
            if len(segmentation_files) == 1:
                segmentation_nii_file = os.path.join(segmentation_folder, segmentation_files[0])

            if ct_nii_file is None or pet_nii_file is None or segmentation_nii_file is None:
                # If any of the NIfTI files is missing, skip this patient
                continue

            # Load NIfTI files
            nifti_ct = nib.load(ct_nii_file)
            nifti_pet = nib.load(pet_nii_file)
            nifti_segmentation = nib.load(segmentation_nii_file)

            # Get voxel size for PET NIfTI
            voxel_size_pet = nifti_pet.header.get_zooms()

            # Resample CT to match PET shape
            ct_data = nifti_ct.get_fdata()
            ct_shape = ct_data.shape
            pet_shape = nifti_pet.get_fdata().shape

            if ct_shape != pet_shape:
                zoom_factors = np.array(pet_shape) / np.array(ct_shape)
                ct_data_resampled = zoom(ct_data, zoom_factors, order=1)
            else:
                ct_data_resampled = ct_data

            # Rotate CT 180 degrees clockwise (two times)
            ct_data_rotated = rotate(ct_data_resampled, angle=180, axes=(1, 0), reshape=False)

            # Save the rotated CT NIfTI with PET voxel size
            new_ct_nii_file = os.path.join(ct_folder, "resampled_and_rotated_ct.nii.gz")
            nib.save(nib.Nifti1Image(ct_data_rotated, nifti_pet.affine, header=nifti_pet.header), new_ct_nii_file)

            # Get the updated CT NIfTI and create a new header with voxel size (1, 1, 1)
            updated_ct_nifti = nib.load(new_ct_nii_file)
            updated_ct_nifti.header.set_zooms((1, 1, 1))

            # Save the CT NIfTI with updated voxel size
            nib.save(updated_ct_nifti, new_ct_nii_file)

            # Set voxel size of PET and Segmentation NIfTI files to (1, 1, 1) and save
            updated_pet_nii = nib.Nifti1Image(nifti_pet.get_fdata(), nifti_pet.affine, header=nifti_pet.header)
            updated_pet_nii.header.set_zooms((1, 1, 1))
            nib.save(updated_pet_nii, pet_nii_file)

            updated_segmentation_nii = nib.Nifti1Image(nifti_segmentation.get_fdata(), nifti_segmentation.affine, header=nifti_segmentation.header)
            updated_segmentation_nii.header.set_zooms((1, 1, 1))
            nib.save(updated_segmentation_nii, segmentation_nii_file)

            print(f"Resampled and rotated CT NIfTI saved to: {new_ct_nii_file}")
            print(f"Updated PET NIfTI saved to: {pet_nii_file}")
            print(f"Updated Segmentation NIfTI saved to: {segmentation_nii_file}")

            # Print voxel size after modification
            print(f"Patient ID: {os.path.basename(root)}")
            print(f"Voxel size of CT NIfTI: {updated_ct_nifti.header.get_zooms()}")
            print(f"Voxel size of PET NIfTI: {updated_pet_nii.header.get_zooms()}")
            print(f"Voxel size of Segmentation NIfTI: {updated_segmentation_nii.header.get_zooms()}")
            print("----------------------------")

if __name__ == "__main__":
    root_folder = '/media/lito/LaCie/CT-TEP_Data/'
    convert(root_folder)
    resample(root_folder)