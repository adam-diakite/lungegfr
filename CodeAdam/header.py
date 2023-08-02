import nibabel as nib
import matplotlib.pyplot as plt
from ipywidgets import interact
import numpy as np
from matplotlib.widgets import Slider
# Example usage:
ct_nii_file = "/media/adamdiakite/LaCie/CT-TEP_Data/2-21-0004/Images/CTnii/2_body-low_dose_ct.nii.gz"
pet_nii_file = "/media/adamdiakite/LaCie/CT-TEP_Data/2-21-0004/Images/PETnii/2-21-0004_pet_float32_SUVbw.nii.gz"
segmentation_nii_file = "/media/adamdiakite/LaCie/CT-TEP_Data/2-21-0004/segmentation/PRIMITIF_PULM_Abs_thres4.0to999.0.uint16.nii.gz"
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



def display_nifti_slices(ct_nii_file, pet_nii_file, segmentation_nii_file):
    # Load NIfTI files
    nifti_ct = nib.load(ct_nii_file)
    nifti_pet = nib.load(pet_nii_file)
    nifti_segmentation = nib.load(segmentation_nii_file)

    # Get image data
    ct_data = nifti_ct.get_fdata()
    pet_data = nifti_pet.get_fdata()
    segmentation_data = nifti_segmentation.get_fdata()

    # Create the figure and subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # Display the CT image
    ct_plot = axes[0, 0].imshow(ct_data[:, :, 0], cmap='gray')
    axes[0, 0].set_title("CT")

    # Display the PET image
    pet_plot = axes[0, 1].imshow(pet_data[:, :, 0], cmap='hot')
    axes[0, 1].set_title("PET")

    # Display the Segmentation image
    seg_plot = axes[1, 0].imshow(segmentation_data[:, :, 0], cmap='binary')
    axes[1, 0].set_title("Segmentation")

    # Calculate the overlay image (CT + PET + Segmentation)
    overlay_data = ct_data + pet_data + segmentation_data
    overlay_plot = axes[1, 1].imshow(overlay_data[:, :, 0], cmap='gray')
    axes[1, 1].set_title("Overlay")

    # Slider settings
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slice_slider = Slider(ax_slider, 'Slice', 0, ct_data.shape[2] - 1, valinit=0, valstep=1)

    def update(val):
        # Get the current slice index from the slider
        slice_index = int(slice_slider.val)

        # Update the displayed images with the selected slice
        ct_plot.set_data(ct_data[:, :, slice_index])
        pet_plot.set_data(pet_data[:, :, slice_index])
        seg_plot.set_data(segmentation_data[:, :, slice_index])
        overlay_data = ct_data + pet_data + segmentation_data
        overlay_plot.set_data(overlay_data[:, :, slice_index])

        # Update the plot
        fig.canvas.draw_idle()

    # Attach the update function to the slider
    slice_slider.on_changed(update)

    # Show the plot
    plt.show()


display_nifti_slices(ct_nii_file, pet_nii_file, segmentation_nii_file)
