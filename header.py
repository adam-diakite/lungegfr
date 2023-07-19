import nibabel as nib
import matplotlib.pyplot as plt
from ipywidgets import interact
import numpy as np

# Example usage:
ct_nii_file = "/media/lito/LaCie/CT-TEP_Data/2-21-0003/Images/CTnii/resampled_and_rotated_ct.nii.gz"
pet_nii_file = "/media/lito/LaCie/CT-TEP_Data/2-21-0003/Images/PETnii/2-21-0003_pet_float32_SUVbw.nii.gz"
segmentation_nii_file = "/media/lito/LaCie/CT-TEP_Data/2-21-0003/segmentation/PRIMITIF_PULM_Abs_thres4.0to999.0.uint16.nii.gz"
def print_header(nii_file):
    nifti_data = nib.load(nii_file)
    header = nifti_data.header
    dimensions = header.get_data_shape()
    print(header)

print(f"CT NIfTI Header:")
print_header(ct_nii_file)
print(f"\nPET NIfTI Header:")
print_header(pet_nii_file)
print(f"\nSegmentation NIfTI Header:")
print_header(segmentation_nii_file)


