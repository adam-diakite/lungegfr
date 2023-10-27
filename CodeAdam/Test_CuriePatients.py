from __future__ import print_function
import numpy as np
import os
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import pandas as pd
# Add the following line to the top of your code
from itertools import combinations
import itertools
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import nibabel as nib
from scipy.stats import iqr

folder_TKI = "/media/adamdiakite/LaCie/CT-TEP_TKI/NPY"
results_TKI = "/media/adamdiakite/LaCie/CT-TEP_TKI/Results"

folder_ICI = "/media/adamdiakite/LaCie/CT-TEP_ICI/NPY"
results_ICI = "/media/adamdiakite/LaCie/CT-TEP_ICI/Results"

csv_TKI = "/media/adamdiakite/LaCie/CT-TEP_TKI/Results/PrecisionPredict_Paris.csv"
csv_ICI = "/media/adamdiakite/LaCie/CT-TEP_ICI/Results/ICI.csv"

lungegfr = load_model(
    '/home/adamdiakite/Documents/lungegfr-master/model/LungEGFR.hdf5')  # ,
lungIO = load_model('/home/adamdiakite/Documents/lungegfr-master/model/LungIO.hdf5')


def load_and_predict_global_average(root_folder, output_folder, model):
    patient_info_list = []

    for patient_folder in os.listdir(root_folder):
        patient_folder_path = os.path.join(root_folder, patient_folder)

        if os.path.isdir(patient_folder_path):
            print(f"Processing patient folder: {patient_folder}")

            ct_npy_path = os.path.join(patient_folder_path, "ct.npy")
            pet_npy_path = os.path.join(patient_folder_path, "pet.npy")
            fuse_npy_path = os.path.join(patient_folder_path, "fuse.npy")
            label_npy_path = os.path.join(patient_folder_path, "label.npy")
            tumor_area_csv_path = os.path.join(patient_folder_path, "tumor_areas.csv")

            if not (os.path.exists(ct_npy_path) and os.path.exists(pet_npy_path)
                    and os.path.exists(fuse_npy_path) and os.path.exists(label_npy_path)
                    and os.path.exists(tumor_area_csv_path)):
                print(f"Skipping patient folder {patient_folder}. NPY or CSV files are missing.")
                continue

            ct_npy = np.load(ct_npy_path)
            pet_npy = np.load(pet_npy_path)
            fuse_npy = np.load(fuse_npy_path)
            label_npy = np.load(label_npy_path)

            ct_data = np.asarray(ct_npy, dtype="float32")
            pet_data = np.asarray(pet_npy, dtype="float32")
            fuse_data = np.asarray(fuse_npy, dtype="float32")
            label_data = np.asarray(label_npy, dtype="float32")

            ct_data = np.expand_dims(ct_data, axis=3)
            pet_data = np.expand_dims(pet_data, axis=3)
            fuse_data = np.expand_dims(fuse_data, axis=3)

            x_data = np.concatenate((pet_data, ct_data, fuse_data), axis=3)
            predictions = model.predict(x_data, verbose=1)

            # Check for NaN predictions
            if np.any(np.isnan(predictions)):
                print(f"Skipping patient folder {patient_folder}. Predictions contain NaN values.")
                continue

            # Save predictions
            patient_output_folder = os.path.join(output_folder, patient_folder)
            os.makedirs(patient_output_folder, exist_ok=True)
            np.savetxt(os.path.join(patient_output_folder, 'predict.txt'), predictions)

            # Calculate average score for each patient (ignoring NaN values)
            average_score = np.nanmean(predictions[:, 1])

            # Calculate the median score among all slice scores (ignoring NaN values)
            median_score_slices = np.nanmedian(predictions[:, 1])

            # Load tumor area from CSV
            tumor_area_df = pd.read_csv(tumor_area_csv_path)

            # Add patient information to the list
            patient_info = {
                "patient_folder": patient_folder,
                "average_score": average_score,
            }
            patient_info_list.append(patient_info)

    # Create a list of average scores
    average_scores = [info["average_score"] for info in patient_info_list]

    # Create a vertical box plot
    plt.figure(figsize=(6, 8))  # Adjust the figure size for vertical orientation
    plt.boxplot(average_scores, vert=True)
    plt.ylabel("Average Score")
    plt.title("Distribution of Average Scores (IQR, Quartiles, Median)")
    plt.show()

    return patient_info_list


def load_and_predict_median(root_folder, output_folder, model):
    patient_info_list = []

    for patient_folder in os.listdir(root_folder):
        patient_folder_path = os.path.join(root_folder, patient_folder)

        if os.path.isdir(patient_folder_path):
            print(f"Processing patient folder: {patient_folder}")

            ct_npy_path = os.path.join(patient_folder_path, "ct.npy")
            pet_npy_path = os.path.join(patient_folder_path, "pet.npy")
            fuse_npy_path = os.path.join(patient_folder_path, "fuse.npy")
            label_npy_path = os.path.join(patient_folder_path, "label.npy")

            if not (os.path.exists(ct_npy_path) and os.path.exists(pet_npy_path)
                    and os.path.exists(fuse_npy_path) and os.path.exists(label_npy_path)):
                print(f"Skipping patient folder {patient_folder}. NPY files are missing.")
                continue

            ct_npy = np.load(ct_npy_path)
            pet_npy = np.load(pet_npy_path)
            fuse_npy = np.load(fuse_npy_path)
            label_npy = np.load(label_npy_path)

            ct_data = np.asarray(ct_npy, dtype="float32")
            pet_data = np.asarray(pet_npy, dtype="float32")
            fuse_data = np.asarray(fuse_npy, dtype="float32")
            label_data = np.asarray(label_npy, dtype="float32")

            ct_data = np.expand_dims(ct_data, axis=3)
            pet_data = np.expand_dims(pet_data, axis=3)
            fuse_data = np.expand_dims(fuse_data, axis=3)

            x_data = np.concatenate((pet_data, ct_data, fuse_data), axis=3)
            predictions = model.predict(x_data, verbose=1)

            # Save predictions
            patient_output_folder = os.path.join(output_folder, patient_folder)
            os.makedirs(patient_output_folder, exist_ok=True)
            np.savetxt(os.path.join(patient_output_folder, 'predict.txt'), predictions)

            # Calculate median score for each patient
            central_slice_index = len(predictions) // 2
            median_score = predictions[central_slice_index, 0]

            # Add patient information to the list
            patient_info = {
                "patient_folder": patient_folder,
                "median_score": median_score
            }
            patient_info_list.append(patient_info)

            print(f"Median score for patient {patient_folder}: {median_score}")

    # Return patient information list
    return patient_info_list


def load_and_predict_quartiles(root_folder, output_folder, model):
    patient_info_list = []

    for patient_folder in os.listdir(root_folder):
        patient_folder_path = os.path.join(root_folder, patient_folder)

        if os.path.isdir(patient_folder_path):
            print(f"Processing patient folder: {patient_folder}")

            ct_npy_path = os.path.join(patient_folder_path, "ct.npy")
            pet_npy_path = os.path.join(patient_folder_path, "pet.npy")
            fuse_npy_path = os.path.join(patient_folder_path, "fuse.npy")
            label_npy_path = os.path.join(patient_folder_path, "label.npy")

            if not (os.path.exists(ct_npy_path) and os.path.exists(pet_npy_path)
                    and os.path.exists(fuse_npy_path) and os.path.exists(label_npy_path)):
                print(f"Skipping patient folder {patient_folder}. NPY files are missing.")
                continue

            ct_npy = np.load(ct_npy_path)
            pet_npy = np.load(pet_npy_path)
            fuse_npy = np.load(fuse_npy_path)
            label_npy = np.load(label_npy_path)

            ct_data = np.asarray(ct_npy, dtype="float32")
            pet_data = np.asarray(pet_npy, dtype="float32")
            fuse_data = np.asarray(fuse_npy, dtype="float32")
            label_data = np.asarray(label_npy, dtype="float32")

            ct_data = np.expand_dims(ct_data, axis=3)
            pet_data = np.expand_dims(pet_data, axis=3)
            fuse_data = np.expand_dims(fuse_data, axis=3)

            x_data = np.concatenate((pet_data, ct_data, fuse_data), axis=3)
            predictions = model.predict(x_data, verbose=1)

            # Save predictions
            patient_output_folder = os.path.join(output_folder, patient_folder)
            os.makedirs(patient_output_folder, exist_ok=True)
            np.savetxt(os.path.join(patient_output_folder, 'predict.txt'), predictions)

            # Calculate mean between Q1 and Q3 of the scores for each patient
            q1 = np.percentile(predictions[:, 0], 25)
            q3 = np.percentile(predictions[:, 0], 75)
            mean_quartiles = (q1 + q3) / 2.0

            # Add patient information to the list
            patient_info = {
                "patient_folder": patient_folder,
                "mean_quartiles_score": mean_quartiles
            }
            patient_info_list.append(patient_info)

            print(f"Mean score between Q1 and Q3 for patient {patient_folder}: {mean_quartiles}")

    # Return patient information list
    return patient_info_list


def load_and_predict_30(root_folder, output_folder, model):
    patient_info_list = []

    for patient_folder in os.listdir(root_folder):
        patient_folder_path = os.path.join(root_folder, patient_folder)

        if os.path.isdir(patient_folder_path):
            print(f"Processing patient folder: {patient_folder}")

            ct_npy_path = os.path.join(patient_folder_path, "ct.npy")
            pet_npy_path = os.path.join(patient_folder_path, "pet.npy")
            fuse_npy_path = os.path.join(patient_folder_path, "fuse.npy")
            label_npy_path = os.path.join(patient_folder_path, "label.npy")
            tumor_area_csv_path = os.path.join(patient_folder_path, "tumor_areas.csv")

            if not (os.path.exists(ct_npy_path) and os.path.exists(pet_npy_path)
                    and os.path.exists(fuse_npy_path) and os.path.exists(label_npy_path)
                    and os.path.exists(tumor_area_csv_path)):
                print(f"Skipping patient folder {patient_folder}. NPY or CSV files are missing.")
                continue

            ct_npy = np.load(ct_npy_path)
            pet_npy = np.load(pet_npy_path)
            fuse_npy = np.load(fuse_npy_path)
            label_npy = np.load(label_npy_path)

            ct_data = np.asarray(ct_npy, dtype="float32")
            pet_data = np.asarray(pet_npy, dtype="float32")
            fuse_data = np.asarray(fuse_npy, dtype="float32")
            label_data = np.asarray(label_npy, dtype="float32")

            ct_data = np.expand_dims(ct_data, axis=3)
            pet_data = np.expand_dims(pet_data, axis=3)
            fuse_data = np.expand_dims(fuse_data, axis=3)

            x_data = np.concatenate((pet_data, ct_data, fuse_data), axis=3)
            predictions = model.predict(x_data, verbose=1)

            # Check for NaN predictions
            if np.any(np.isnan(predictions)):
                print(f"Skipping patient folder {patient_folder}. Predictions contain NaN values.")
                continue

            # Save predictions
            patient_output_folder = os.path.join(output_folder, patient_folder)
            os.makedirs(patient_output_folder, exist_ok=True)
            np.savetxt(os.path.join(patient_output_folder, 'predict.txt'), predictions)

            # Calculate average score for each patient (ignoring NaN values)
            average_score = np.nanmean(predictions[:, 1])

            # Calculate the median score among all slice scores (ignoring NaN values)
            median_score_slices = np.nanmedian(predictions[:, 1])

            # Load tumor area from CSV
            tumor_area_df = pd.read_csv(tumor_area_csv_path)

            # Find the maximum tumor area
            max_tumor_area = tumor_area_df['Tumor Area'].max()

            # Select tumor areas with at least 30% of the maximum
            selected_slices = tumor_area_df[tumor_area_df['Tumor Area'] >= 0.3 * max_tumor_area]

            # Get the indices of the selected slices
            selected_indices = selected_slices.index.to_list()

            # # Filter slices to compute based on the selected indices
            # ct_data = ct_data[:, :, selected_indices]
            # pet_data = pet_data[:, :, selected_indices]
            # fuse_data = fuse_data[:, :, selected_indices]
            # label_data = label_data[:, :, selected_indices]

            # Print the selected slice numbers for the patient
            print(f"Selected {len(selected_indices)} slices for patient {patient_folder}:")
            print(selected_slices)

            # Perform the rest of the computation with the selected slices...

            # Add patient information to the list
            patient_info = {
                "patient_folder": patient_folder,
                "average_score": average_score,
            }
            patient_info_list.append(patient_info)

    # Calculate the median score among all average scores (ignoring NaN values)
    median_score_average = np.nanmedian([info["average_score"] for info in patient_info_list])

    # Calculate the Interquartile Range (IQR) with the average score (ignoring NaN values)
    iqr_score = iqr([info["average_score"] for info in patient_info_list], nan_policy='omit')

    # Print general information across all patients
    print(f"\nGeneral Information Across All Patients:")
    print(f"Average score: {np.nanmean([info['average_score'] for info in patient_info_list])}")
    print(f"Median score among all slice scores: {median_score_slices}")
    print(f"Median score among all average scores: {median_score_average}")
    print(f"IQR with the average score: {iqr_score}\n")

    return patient_info_list


def load_and_predict_30(root_folder, output_folder, model):
    patient_info_list = []

    for patient_folder in os.listdir(root_folder):
        patient_folder_path = os.path.join(root_folder, patient_folder)

        if os.path.isdir(patient_folder_path):
            print(f"Processing patient folder: {patient_folder}")

            ct_npy_path = os.path.join(patient_folder_path, "ct.npy")
            pet_npy_path = os.path.join(patient_folder_path, "pet.npy")
            fuse_npy_path = os.path.join(patient_folder_path, "fuse.npy")
            label_npy_path = os.path.join(patient_folder_path, "label.npy")
            tumor_area_csv_path = os.path.join(patient_folder_path, "tumor_areas.csv")

            if not (os.path.exists(ct_npy_path) and os.path.exists(pet_npy_path)
                    and os.path.exists(fuse_npy_path) and os.path.exists(label_npy_path)
                    and os.path.exists(tumor_area_csv_path)):
                print(f"Skipping patient folder {patient_folder}. NPY or CSV files are missing.")
                continue

            ct_npy = np.load(ct_npy_path)
            pet_npy = np.load(pet_npy_path)
            fuse_npy = np.load(fuse_npy_path)
            label_npy = np.load(label_npy_path)

            ct_data = np.asarray(ct_npy, dtype="float32")
            pet_data = np.asarray(pet_npy, dtype="float32")
            fuse_data = np.asarray(fuse_npy, dtype="float32")
            label_data = np.asarray(label_npy, dtype="float32")

            ct_data = np.expand_dims(ct_data, axis=3)
            pet_data = np.expand_dims(pet_data, axis=3)
            fuse_data = np.expand_dims(fuse_data, axis=3)

            x_data = np.concatenate((pet_data, ct_data, fuse_data), axis=3)
            predictions = model.predict(x_data, verbose=1)

            # Check for NaN predictions
            if np.any(np.isnan(predictions)):
                print(f"Skipping patient folder {patient_folder}. Predictions contain NaN values.")
                continue

            # Save predictions
            patient_output_folder = os.path.join(output_folder, patient_folder)
            os.makedirs(patient_output_folder, exist_ok=True)
            np.savetxt(os.path.join(patient_output_folder, 'predict.txt'), predictions)

            # Calculate average score for each patient (ignoring NaN values)
            average_score = np.nanmean(predictions[:, 1])

            # Calculate the median score among all slice scores (ignoring NaN values)
            median_score_slices = np.nanmedian(predictions[:, 1])

            # Load tumor area from CSV
            tumor_area_df = pd.read_csv(tumor_area_csv_path)

            # Find the maximum tumor area
            max_tumor_area = tumor_area_df['Tumor Area'].max()

            # Select tumor areas with at least 30% of the maximum
            selected_slices = tumor_area_df[tumor_area_df['Tumor Area'] >= 0.3 * max_tumor_area]

            # Get the indices of the selected slices
            selected_indices = selected_slices.index.to_list()

            # # Filter slices to compute based on the selected indices
            # ct_data = ct_data[:, :, selected_indices]
            # pet_data = pet_data[:, :, selected_indices]
            # fuse_data = fuse_data[:, :, selected_indices]
            # label_data = label_data[:, :, selected_indices]

            # Print the selected slice numbers for the patient
            print(f"Selected {len(selected_indices)} slices for patient {patient_folder}:")
            print(selected_slices)

            # Perform the rest of the computation with the selected slices...

            # Add patient information to the list
            patient_info = {
                "patient_folder": patient_folder,
                "average_score": average_score,
            }
            patient_info_list.append(patient_info)

    # Calculate the median score among all average scores (ignoring NaN values)
    median_score_average = np.nanmedian([info["average_score"] for info in patient_info_list])

    # Calculate the Interquartile Range (IQR) with the average score (ignoring NaN values)
    iqr_score = iqr([info["average_score"] for info in patient_info_list], nan_policy='omit')

    # Collect all average scores
    all_average_scores = [info["average_score"] for info in patient_info_list if not np.isnan(info["average_score"])]

    # Print general information across all patients
    print(f"\nGeneral Information Across All Patients:")
    print(f"Average score: {np.nanmean(all_average_scores)}")
    print(f"Median score among all slice scores: {median_score_slices}")
    print(f"Median score among all average scores: {median_score_average}")
    print(f"IQR with the average score: {iqr_score}\n")

    # Create a boxplot of the average scores
    plt.figure(figsize=(8, 6))
    plt.boxplot(all_average_scores, vert=True)
    plt.title("Scores distribution for TKI patients")
    plt.xlabel("Scores")
    plt.show()

    return patient_info_list

def load_and_predict_30_comp(root_folder1, root_folder2, output_folder1, output_folder2, model):
    patient_info_list1 = []
    patient_info_list2 = []

    for root_folder, output_folder, patient_info_list in [(root_folder1, output_folder1, patient_info_list1),
                                                          (root_folder2, output_folder2, patient_info_list2)]:
        for patient_folder in os.listdir(root_folder):
            patient_folder_path = os.path.join(root_folder, patient_folder)

            if os.path.isdir(patient_folder_path):
                print(f"Processing patient folder: {patient_folder}")

                ct_npy_path = os.path.join(patient_folder_path, "ct.npy")
                pet_npy_path = os.path.join(patient_folder_path, "pet.npy")
                fuse_npy_path = os.path.join(patient_folder_path, "fuse.npy")
                label_npy_path = os.path.join(patient_folder_path, "label.npy")
                tumor_area_csv_path = os.path.join(patient_folder_path, "tumor_areas.csv")

                if not (os.path.exists(ct_npy_path) and os.path.exists(pet_npy_path)
                        and os.path.exists(fuse_npy_path) and os.path.exists(label_npy_path)
                        and os.path.exists(tumor_area_csv_path)):
                    print(f"Skipping patient folder {patient_folder}. NPY or CSV files are missing.")
                    continue

                ct_npy = np.load(ct_npy_path)
                pet_npy = np.load(pet_npy_path)
                fuse_npy = np.load(fuse_npy_path)
                label_npy = np.load(label_npy_path)

                ct_data = np.asarray(ct_npy, dtype="float32")
                pet_data = np.asarray(pet_npy, dtype="float32")
                fuse_data = np.asarray(fuse_npy, dtype="float32")
                label_data = np.asarray(label_npy, dtype="float32")

                ct_data = np.expand_dims(ct_data, axis=3)
                pet_data = np.expand_dims(pet_data, axis=3)
                fuse_data = np.expand_dims(fuse_data, axis=3)

                x_data = np.concatenate((pet_data, ct_data, fuse_data), axis=3)
                predictions = model.predict(x_data, verbose=1)

                # Check for NaN predictions
                if np.any(np.isnan(predictions)):
                    print(f"Skipping patient folder {patient_folder}. Predictions contain NaN values.")
                    continue

                # Save predictions
                patient_output_folder = os.path.join(output_folder, patient_folder)
                os.makedirs(patient_output_folder, exist_ok=True)
                np.savetxt(os.path.join(patient_output_folder, 'predict.txt'), predictions)

                # Calculate average score for each patient (ignoring NaN values)
                average_score = np.nanmean(predictions[:, 1])

                # Calculate the median score among all slice scores (ignoring NaN values)
                median_score_slices = np.nanmedian(predictions[:, 1])

                # Load tumor area from CSV
                tumor_area_df = pd.read_csv(tumor_area_csv_path)

                # Find the maximum tumor area
                max_tumor_area = tumor_area_df['Tumor Area'].max()

                # Select tumor areas with at least 30% of the maximum
                selected_slices = tumor_area_df[tumor_area_df['Tumor Area'] >= 0.3 * max_tumor_area]

                # Get the indices of the selected slices
                selected_indices = selected_slices.index.to_list()

                # # Filter slices to compute based on the selected indices
                # ct_data = ct_data[:, :, selected_indices]
                # pet_data = pet_data[:, :, selected_indices]
                # fuse_data = fuse_data[:, :, selected_indices]
                # label_data = label_data[:, :, selected_indices]

                # Print the selected slice numbers for the patient
                print(f"Selected {len(selected_indices)} slices for patient {patient_folder}:")
                print(selected_slices)

                # Perform the rest of the computation with the selected slices...

                # Add patient information to the list
                patient_info = {
                    "patient_folder": patient_folder,
                    "average_score": average_score,
                }
                patient_info_list.append(patient_info)

    # Calculate the median score among all average scores (ignoring NaN values)
    median_score_average1 = np.nanmedian([info["average_score"] for info in patient_info_list1])
    median_score_average2 = np.nanmedian([info["average_score"] for info in patient_info_list2])

    # Calculate the Interquartile Range (IQR) with the average score (ignoring NaN values)
    iqr_score1 = iqr([info["average_score"] for info in patient_info_list1], nan_policy='omit')
    iqr_score2 = iqr([info["average_score"] for info in patient_info_list2], nan_policy='omit')

    # Collect all average scores
    all_average_scores1 = [info["average_score"] for info in patient_info_list1 if not np.isnan(info["average_score"])]
    all_average_scores2 = [info["average_score"] for info in patient_info_list2 if not np.isnan(info["average_score"])]

    # Print general information across all patients for both folders
    print(f"\nGeneral Information for TKI Patients:")
    print(f"Average score: {np.nanmean(all_average_scores1)}")
    print(f"Median score among all average scores: {median_score_average1}")
    print(f"IQR with the average score: {iqr_score1}\n")

    print(f"\nGeneral Information for ICI Patients:")
    print(f"Average score: {np.nanmean(all_average_scores2)}")
    print(f"Median score among all average scores: {median_score_average2}")
    print(f"IQR with the average score: {iqr_score2}\n")

    # Create a single figure with two boxplots
    plt.figure(figsize=(12, 6))

    # Boxplot for Folder 1
    plt.subplot(1, 2, 1)
    plt.boxplot(all_average_scores1, vert=True)
    plt.title("Scores distribution for TKI Patients")
    plt.xlabel("Scores")

    # Boxplot for Folder 2
    plt.subplot(1, 2, 2)
    plt.boxplot(all_average_scores2, vert=True)
    plt.title("Scores distribution for ICI Patients")
    plt.xlabel("Scores")

    plt.tight_layout()
    plt.show()

    return patient_info_list1, patient_info_list2


def cox_ph_model(patient_info_list):
    # Create a DataFrame from the patient information list
    df = pd.DataFrame(patient_info_list)

    # Drop the 'patient_folder' column
    df = df.drop(columns=['patient_folder'])

    # Fit the Cox Proportional Hazards model
    cph = CoxPHFitter()
    cph.fit(df, duration_col='Time', event_col='Status')

    # Plot the model
    cph.plot()

    # Print the model summary
    print(cph.summary)

    return cph


def load_additional_data(csv_file, patient_info_list):
    # Load additional data from the CSV file
    df = pd.read_csv(csv_file)

    # Create a dictionary to map patient IDs to additional data
    additional_data_dict = {}
    for index, row in df.iterrows():
        additional_data_dict[row['ID_patient']] = row.to_dict()

    # Combine the patient_info_list with additional data (if available)
    patient_info_list_with_data = []
    for patient_info in patient_info_list:
        patient_id = patient_info['patient_folder']
        if patient_id in additional_data_dict:
            patient_info.update(additional_data_dict[patient_id])
            patient_info_list_with_data.append(patient_info)
        else:
            print(f"Additional data not found for patient ID: {patient_id}")

    return patient_info_list_with_data


def kaplan_meier_analysis(patient_info_list, threshold=0.5):
    # Create a DataFrame from the patient information list
    df = pd.DataFrame(patient_info_list)

    # Drop the 'patient_folder' column
    df = df.drop(columns=['patient_folder'])

    # Set the threshold value to separate patients into two groups
    df['Group'] = df['average_score'].apply(lambda x: 'High DLS' if x >= threshold else 'Low DLS')

    # Initialize the Kaplan-Meier estimator for high DLS group
    kmf_high_dls = KaplanMeierFitter()
    kmf_high_dls.fit(df[df['Group'] == 'High DLS']['Time'], event_observed=df[df['Group'] == 'High DLS']['Status'],
                     label='High DLS')

    # Initialize the Kaplan-Meier estimator for low DLS group
    kmf_low_dls = KaplanMeierFitter()
    kmf_low_dls.fit(df[df['Group'] == 'Low DLS']['Time'], event_observed=df[df['Group'] == 'Low DLS']['Status'],
                    label='Low DLS')

    # Perform the log-rank test to calculate the p-value between the two groups
    results = logrank_test(df[df['Group'] == 'High DLS']['Time'], df[df['Group'] == 'Low DLS']['Time'],
                           event_observed_A=df[df['Group'] == 'High DLS']['Status'],
                           event_observed_B=df[df['Group'] == 'Low DLS']['Status'])

    # Get the threshold value for the title
    threshold_str = f'Threshold: {threshold:.2f}'

    # Create a string for the p-value to include in the title
    p_value_str = f'P-Value: {results.p_value:.6f}'

    # Combine the titles with the threshold value and p-value
    plt_title = f'Kaplan-Meier Survival Curve ({threshold_str}) {p_value_str}'

    # Plot the Kaplan-Meier survival curves for each group with separate colors
    plt.figure(figsize=(10, 6))
    kmf_high_dls.plot(color='blue')
    kmf_low_dls.plot(color='red')
    plt.title(plt_title)
    plt.xlabel('Time (months)')
    plt.ylabel('Survival Probability')
    plt.grid()
    plt.legend(loc='lower left')
    plt.show()

    return kmf_high_dls, kmf_low_dls


def optimal_kaplan_meier(patient_info_list):
    # Sort the patient_info_list based on 'mean_quartiles_score'
    sorted_list = sorted(patient_info_list, key=lambda x: x['average_score'])

    # Get the observed times and events for all patients
    all_times = []
    all_events = []
    for patient in sorted_list:
        if 'Time' in patient and 'Status' in patient:
            all_times.append(patient['Time'])
            all_events.append(patient['Status'])

    all_times = np.array(all_times)
    all_events = np.array(all_events)

    # Get the number of patients
    num_patients = len(sorted_list)

    # Initialize variables to keep track of the optimal group configuration
    min_p_value = float('inf')
    optimal_group1 = []
    optimal_group2 = []

    # Iterate over different group sizes
    for group1_size in range(1, num_patients):
        # Check if the group size is less than 5 for both groups
        if group1_size >= 5 and num_patients - group1_size >= 5:
            # Get the indices for group 1 and group 2
            group1_indices = list(range(group1_size))
            group2_indices = list(range(group1_size, num_patients))

            # Get the times and events for group 1 and group 2
            group1_times = all_times[group1_indices]
            group1_events = all_events[group1_indices]
            group2_times = all_times[group2_indices]
            group2_events = all_events[group2_indices]

            # Perform log-rank test to calculate the p-value
            results = logrank_test(group1_times, group2_times, event_observed_A=group1_events,
                                   event_observed_B=group2_events)
            p_value = results.p_value

            # Update the optimal group configuration if the p-value is smaller
            if p_value < min_p_value:
                min_p_value = p_value
                optimal_group1 = group1_indices
                optimal_group2 = group2_indices

    # Get the patient information for the optimal groups
    optimal_group1_info = [sorted_list[i] for i in optimal_group1]
    optimal_group2_info = [sorted_list[i] for i in optimal_group2]

    # Print the optimal group configuration and p-value
    print("\nOptimal Low Scores Group:")
    for patient_info in optimal_group1_info:
        print(f"Patient ID: {patient_info['patient_folder']}, DLS Score: {patient_info['average_score']}, "
              f"Time: {patient_info.get('Time', 'Not available')}, Status: {patient_info.get('Status', 'Not available')}")

    print("\nOptimal High Scores Group:")
    for patient_info in optimal_group2_info:
        print(f"Patient ID: {patient_info['patient_folder']}, DLS Score: {patient_info['average_score']}, "
              f"Time: {patient_info.get('Time', 'Not available')}, Status: {patient_info.get('Status', 'Not available')}")

    # Perform the log-rank test to calculate the p-value between the two groups
    results = logrank_test(all_times[optimal_group1], all_times[optimal_group2],
                           event_observed_A=all_events[optimal_group1],
                           event_observed_B=all_events[optimal_group2])
    p_value = results.p_value

    # Plot the Kaplan-Meier survival curves for the optimal groups
    kmf_optimal_group1 = KaplanMeierFitter()
    kmf_optimal_group1.fit(all_times[optimal_group1], event_observed=all_events[optimal_group1], label='Low DLS')

    kmf_optimal_group2 = KaplanMeierFitter()
    kmf_optimal_group2.fit(all_times[optimal_group2], event_observed=all_events[optimal_group2], label='High DLS')

    # Plot the Kaplan-Meier survival curves for the optimal groups with separate colors
    plt.figure(figsize=(10, 6))
    kmf_optimal_group1.plot(color='red')
    kmf_optimal_group2.plot(color='blue')

    # Get the p-value for the optimal group configuration
    p_value_str = f"P-Value: {p_value:.6f}"

    # Combine the titles with the p-value
    plt_title = f'Optimal Kaplan-Meier Survival Curve\n{p_value_str}'

    plt.title(plt_title)
    plt.xlabel('Time (months)')
    plt.ylabel('Survival Probability')
    plt.grid()
    plt.legend(loc='lower left')
    plt.show()

    return kmf_optimal_group1, kmf_optimal_group2


# Assuming folder, results, modelpetct1, and csv are defined

patient_info_list = load_and_predict_30_comp(folder_TKI, folder_ICI, results_TKI, results_ICI, lungegfr)
patient_info_list_with_data = load_additional_data(csv_TKI, patient_info_list)
kaplan_meier_analysis(patient_info_list_with_data)




