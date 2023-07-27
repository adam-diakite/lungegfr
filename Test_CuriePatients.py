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

folder = "/media/lito/LaCie/CT-TEP_Data/NPY/"
results = "/media/lito/LaCie/CT-TEP_Data/Results"
csv = "/home/lito/PycharmProjects/lungegfr/Alldata/PrecisionPredict_Paris.csv"

modelpetct1 = load_model(
    '/home/lito/PycharmProjects/lungegfr/model/LungEGFR.hdf5')  # ,weightspatient2-improvement-40-0.67


# modelpetct1.summary()


def load_and_predict(root_folder, output_folder, model):
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

            # Calculate average score for each patient
            average_score = np.mean(predictions[:, 0])

            # Add patient information to the list
            patient_info = {
                "patient_folder": patient_folder,
                "average_score": average_score
            }
            patient_info_list.append(patient_info)

            print(f"Average score for patient {patient_folder}: {average_score}")

    # Return patient information list
    return patient_info_list


def load_additional_data(csv_file, patient_info_list):
    # Load additional data from the CSV file
    additional_data = pd.read_csv(csv_file)

    # Create a dictionary to store the additional data using patient IDs as keys
    additional_data_dict = {}
    for _, row in additional_data.iterrows():
        additional_data_dict[row['ID_patient']] = {
            'Time': row['Time'],
            'Status': row['Status']
        }

    # Connect the additional data to the patient information using patient IDs
    for patient_info in patient_info_list:
        patient_folder = patient_info['patient_folder']
        patient_id = patient_folder  # Assuming the patient folder name is the same as the ID

        if patient_id in additional_data_dict:
            time = additional_data_dict[patient_id]['Time']
            patient_info['Time'] = additional_data_dict[patient_id]['Time']
            patient_info['Status'] = additional_data_dict[patient_id]['Status']
        else:
            print(f"Additional data not found for patient ID: {patient_id}")

    return patient_info_list


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


def kaplan_meier_analysis(patient_info_list):
    # Create a DataFrame from the patient information list
    df = pd.DataFrame(patient_info_list)

    # Drop the 'patient_folder' column
    df = df.drop(columns=['patient_folder'])

    # Calculate the median of the 'average_score' column
    median_score = df['average_score'].median()

    # Separate patients into two groups based on average_score
    df['Group'] = df['average_score'].apply(lambda x: 'High DLS' if x >= median_score else 'Low DLS')

    # Initialize the Kaplan-Meier estimator for high DLS group
    kmf_high_dls = KaplanMeierFitter()
    kmf_high_dls.fit(df[df['Group'] == 'High DLS']['Time'], event_observed=df[df['Group'] == 'High DLS']['Status'],
                     label='High DLS')

    # Initialize the Kaplan-Meier estimator for low DLS group
    kmf_low_dls = KaplanMeierFitter()
    kmf_low_dls.fit(df[df['Group'] == 'Low DLS']['Time'], event_observed=df[df['Group'] == 'Low DLS']['Status'],
                    label='Low DLS')

    # Plot the Kaplan-Meier survival curves for each group with separate colors
    plt.figure(figsize=(10, 6))
    kmf_high_dls.plot(color='red')
    kmf_low_dls.plot(color='blue')
    plt.title('Kaplan-Meier Survival Curve')
    plt.xlabel('Time (months)')
    plt.ylabel('Survival Probability')
    plt.grid()
    plt.legend(loc='lower left')
    plt.show()

    return kmf_high_dls, kmf_low_dls


def optimal_kaplan_meier(patient_info_list):
    # Sort the patient_info_list based on 'average_score'
    sorted_list = sorted(patient_info_list, key=lambda x: x['average_score'])

    # Create a DataFrame to handle the data
    df = pd.DataFrame(sorted_list)

    # Drop rows with missing values in 'Time' or 'Status'
    df = df.dropna(subset=['Time', 'Status'])

    # Get the observed times and events for all patients
    all_times = df['Time'].values
    all_events = df['Status'].values

    # Get the number of patients
    num_patients = len(all_times)

    # Initialize variables to keep track of the optimal group configuration
    min_p_value = float('inf')
    optimal_group1 = []
    optimal_group2 = []

    # Iterate over different group sizes
    for group1_size in range(1, num_patients):
        # Get the indices for group 1 and group 2
        group1_indices = list(range(group1_size))
        group2_indices = list(range(group1_size, num_patients))

        # Get the times and events for group 1 and group 2
        group1_times = all_times[group1_indices]
        group1_events = all_events[group1_indices]
        group2_times = all_times[group2_indices]
        group2_events = all_events[group2_indices]

        # Perform log-rank test to calculate the p-value
        results = logrank_test(group1_times, group2_times, event_observed_A=group1_events, event_observed_B=group2_events)
        p_value = results.p_value

        # Get the average scores for group 1 and group 2
        group1_scores = df['average_score'].iloc[group1_indices].values
        group2_scores = df['average_score'].iloc[group2_indices].values

        # Print the current group configuration, the associated p-value, and the average scores
        # print(f"Group 1 Size: {group1_size}, P-Value: {p_value:.6f}")
        # print(f"Group 1 Scores: {group1_scores}")
        # print(f"Group 2 Scores: {group2_scores}")

        # Update the optimal group configuration if the p-value is smaller
        if p_value < min_p_value:
            min_p_value = p_value
            optimal_group1 = group1_indices
            optimal_group2 = group2_indices

    # Get the patient information for the optimal groups
    optimal_group1_info = df.iloc[optimal_group1]
    optimal_group2_info = df.iloc[optimal_group2]

    # Plot the Kaplan-Meier survival curves for the optimal groups
    kmf_optimal_group1 = KaplanMeierFitter()
    kmf_optimal_group1.fit(all_times[optimal_group1], event_observed=all_events[optimal_group1], label='Optimal Group 1')

    kmf_optimal_group2 = KaplanMeierFitter()
    kmf_optimal_group2.fit(all_times[optimal_group2], event_observed=all_events[optimal_group2], label='Optimal Group 2')

    # Plot the Kaplan-Meier survival curves for the optimal groups with separate colors
    plt.figure(figsize=(10, 6))

    kmf_optimal_group1.plot(color='red')
    kmf_optimal_group2.plot(color='blue')

    plt.title(f'Optimal Kaplan-Meier Survival Curve\nOptimal P-Value: {min_p_value:.6f}')
    plt.xlabel('Time (months)')
    plt.ylabel('Survival Probability')
    plt.grid()
    plt.legend(loc='lower left')
    plt.show()

    # Print the scores for the optimal group
    print("Optimal Group 1 Scores:")
    print(optimal_group1_info["average_score"].values)

    print("\nOptimal Group 2 Scores:")
    print(optimal_group2_info["average_score"].values)



    return kmf_optimal_group1, kmf_optimal_group2


patient_info_list = load_and_predict(folder, results, modelpetct1)
patient_info_list_with_data = load_additional_data(csv, patient_info_list)

# print(patient_info_list)
print(patient_info_list_with_data)

# kmf_high_dls, kmf_low_dls = kaplan_meier_analysis(patient_info_list_with_data)
o_kmf_high_dls, o_mkf_low_dls = optimal_kaplan_meier(patient_info_list_with_data)

# print(patient_info_list_with_data)
# # Printing the combined patient information list
# for patient_info in patient_info_list_with_data:
#     print(f"Patient ID: {patient_info['patient_folder']}, DLS Score: {patient_info['average_score']}, "
#           f"Time: {patient_info.get('Time', 'Not available')}, Status: {patient_info.get('Status', 'Not available')}")
# kaplan = kaplan_meier_analysis(patient_info_list)
# print(kaplan)