import os
import numpy as np
import nibabel as nib
import dicom2nifti
from nilearn.image import resample_to_img, resample_img


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
            print(f"Processing directory: {root}")
            ct_folder = os.path.join(root, "Images", "CTnii")
            pet_folder = os.path.join(root, "Images", "PETnii")
            segmentation_folder = os.path.join(root, "segmentation")

            # Check if the necessary subdirectories exist
            if not os.path.exists(ct_folder) or not os.path.exists(pet_folder) or not os.path.exists(segmentation_folder):
                print("Error: Subdirectories not found in the current directory.")
                continue

            ct_files = os.listdir(ct_folder)
            pet_files = os.listdir(pet_folder)
            segmentation_files = [f for f in os.listdir(segmentation_folder) if f.endswith(".nii.gz")]

            if len(ct_files) == 1 and len(pet_files) == 1 and len(segmentation_files) == 1:
                ct_nii_file = os.path.join(ct_folder, ct_files[0])
                pet_nii_file = os.path.join(pet_folder, pet_files[0])
                segmentation_nii_file = os.path.join(segmentation_folder, segmentation_files[0])

                nifti_ct = nib.load(ct_nii_file)
                nifti_pet = nib.load(pet_nii_file)
                nifti_segmentation = nib.load(segmentation_nii_file)

                # Resample CT to match PET shape and affine
                print("Resampling CT to match PET shape and affine...")
                ct_data_resampled = resample_to_img(nifti_ct, nifti_pet, interpolation="linear").get_fdata()

                # Save the resampled CT NIfTI with PET voxel size in place (overwrite the original CT file)
                print("Saving the resampled CT NIfTI with PET voxel size in place...")
                nib.save(nib.Nifti1Image(ct_data_resampled, nifti_pet.affine, header=nifti_pet.header), ct_nii_file)

                # Update voxel size for PET and Segmentation NIfTI files
                for nifti_file in [pet_nii_file, segmentation_nii_file]:
                    nifti_data = nib.load(nifti_file)
                    nifti_data.header.set_zooms((1, 1, 1))
                    nib.save(nifti_data, nifti_file)

                # Resample CT, PET, and Segmentation to have the same voxel size (1, 1, 1)
                target_affine = np.diag((1, 1, 1))

                # Resample PET
                print("Resampling PET with updated voxel size...")
                resampled_pet = resample_img(nifti_pet, target_affine=target_affine)
                nib.save(resampled_pet, pet_nii_file)

                # Resample Segmentation
                print("Resampling Segmentation with updated voxel size...")
                resampled_segmentation = resample_img(nifti_segmentation, target_affine=target_affine)
                nib.save(resampled_segmentation, segmentation_nii_file)

                # Resample CT again using the same target_affine
                print("Resampling CT with updated voxel size...")
                resampled_ct = resample_img(nifti_ct, target_affine=target_affine)
                nib.save(resampled_ct, ct_nii_file)

                print(f"Resampled CT NIfTI saved to {ct_nii_file}.")
                print(f"Updated PET NIfTI saved to: {pet_nii_file}")
                print(f"Updated Segmentation NIfTI saved to: {segmentation_nii_file}")
            else:
                print("Error: The required number of files not found in the current directory.")

if __name__ == "__main__":
    root_folder = '/media/lito/LaCie/problème/'
    resample(root_folder)


