import nibabel as nib
import numpy as np
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
def calculate_tumor_areas(nifti_file_path, threshold=0.5):
    nifti_segmentation = nib.load(nifti_file_path)
    seg_data = nifti_segmentation.get_fdata()

    max_value = seg_data.max()
    seg_data_normalized = seg_data / max_value

    seg_data_normalized[seg_data_normalized < threshold] = 0
    seg_data_normalized[seg_data_normalized >= threshold] = 1
    print(seg_data_normalized.shape)
    tumor_areas_per_slice = []
    for i in range(seg_data_normalized.shape[-1]):
        slice_data = seg_data_normalized[..., i]
        tumor_area_slice = slice_data.sum()
        tumor_areas_per_slice.append(tumor_area_slice)

    tumor_areas_array = np.array(tumor_areas_per_slice)

    return tumor_areas_array

# Replace 'path_to_tumor_segmentation.nii.gz' with the actual path to your tumor segmentation NIfTI file.
tumor_areas_array = calculate_tumor_areas('/media/adamdiakite/LaCie/CT-TEP_TKI/2-21-0045/segmentation/PRIMITIF_PULM_Abs_thres4.0to999.0.uint16.nii.gz')
print(tumor_areas_array)
