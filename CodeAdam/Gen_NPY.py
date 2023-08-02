import os
import numpy as np
from PIL import Image

folder = "/media/adamdiakite/LaCie/CT-TEP_ICI"
dest = "/media/adamdiakite/LaCie/CT-TEP_ICI/NPY"


def check_png_folders(root_folder):
    """
    Checks for the presence of "CT_PNG" and "PET_PNG" folders in the "Images" folder for each subdirectory.

    :param root_folder: The root directory to start searching.
    """
    all_ok = True  # Flag to track if both folders are present in all subdirectories

    for root, dirs, _ in os.walk(root_folder):
        if "Images" in dirs:
            ct_png_folder = os.path.join(root, "Images", "CT_PNG")
            pet_png_folder = os.path.join(root, "Images", "PET_PNG")

            print(f"Checking folder: {root}")

            ct_ok = os.path.exists(ct_png_folder)
            pet_ok = os.path.exists(pet_png_folder)

            if ct_ok:
                print("CT_PNG folder is present.")
            else:
                print("CT_PNG folder is NOT present.")

            if pet_ok:
                print("PET_PNG folder is present.")
            else:
                print("PET_PNG folder is NOT present.")

            # Update the flag if any folder is missing
            if not (ct_ok and pet_ok):
                all_ok = False

    if all_ok:
        print("Everything is OK.")
    else:
        print("Some folders are missing CT_PNG and/or PET_PNG.")


def normalization_img(image):
    return (image - np.mean(image)) / np.std(image)


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

            if os.path.exists(ct_png_folder) and os.path.exists(pet_png_folder):
                print(f"Processing folders: {ct_png_folder} and {pet_png_folder}")

                ct_files = sorted(os.listdir(ct_png_folder))
                pet_files = sorted(os.listdir(pet_png_folder))

                if len(ct_files) != len(pet_files):
                    print("Mismatch in the number of files in CT_PNG and PET_PNG folders. Skipping this patient.")
                    continue

                num_files = len(ct_files)
                cts = np.empty((num_files, 64, 64), dtype=np.float64)
                pets = np.empty((num_files, 64, 64), dtype=np.float64)
                fuses = np.empty((num_files, 64, 64), dtype=np.float64)

                for i, (ct_file, pet_file) in enumerate(zip(ct_files, pet_files)):
                    ct_path = os.path.join(ct_png_folder, ct_file)
                    pet_path = os.path.join(pet_png_folder, pet_file)

                    ct_image = np.array(Image.open(ct_path), dtype=np.float64)
                    pet_image = np.array(Image.open(pet_path), dtype=np.float64)

                    if len(ct_image.shape) > 2:
                        ct_image = ct_image[:, :, 0]
                    if len(pet_image.shape) > 2:
                        pet_image = pet_image[:, :, 0]

                    ct_image = normalization_img(np.array(Image.fromarray(ct_image).resize((64, 64), Image.NEAREST)))
                    pet_image = normalization_img(np.array(Image.fromarray(pet_image).resize((64, 64), Image.NEAREST)))

                    ct_image = (ct_image - np.mean(ct_image)) / np.std(ct_image)
                    pet_image = (pet_image - np.mean(pet_image)) / np.std(pet_image)
                    fuse_image = pet_image + ct_image
                    fuse_image = (fuse_image - np.mean(fuse_image)) / np.std(fuse_image)

                    cts[i] = ct_image
                    pets[i] = pet_image
                    fuses[i] = fuse_image

                patient_name = os.path.basename(root)
                patient_output_folder = os.path.join(output_folder, patient_name)
                os.makedirs(patient_output_folder, exist_ok=True)

                np.save(os.path.join(patient_output_folder, 'ct.npy'), cts)
                np.save(os.path.join(patient_output_folder, 'pet.npy'), pets)
                np.save(os.path.join(patient_output_folder, 'fuse.npy'), fuses)

                # Create the 'label.npy' file with random 0s and 1s
                label = np.random.randint(2, size=num_files)
                np.save(os.path.join(patient_output_folder, 'label.npy'), label)

                print(f"Arrays and label for patient {patient_name} saved successfully.")
            else:
                print("Both CT_PNG and PET_PNG folders are not present in the current folder.")
                continue


process_png_folders(folder, dest)