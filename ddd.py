
from __future__ import print_function
import numpy as np
import os
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import pandas as pd
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

folder = "/media/adamdiakite/LaCie/CT-TEP_Data/NPY"
results = "/media/adamdiakite/LaCie/CT-TEP_Data/Results"
csv = "/media/adamdiakite/LaCie/CT-TEP_Data/Results/PrecisionPredict_Paris.csv"
modelpetct1 = load_model(
    '/home/adamdiakite/Documents/lungegfr-master/model/LungEGFR.hdf5')  # ,weightspatient2-improvement-40-0.67
modelpetct1.summary()


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


def group_config(kmf_high_dls, kmf_low_dls):
    # Get the time and status data for both groups
    time_high_dls = kmf_high_dls.timeline
    status_high_dls = kmf_high_dls.event_observed
    time_low_dls = kmf_low_dls.timeline
    status_low_dls = kmf_low_dls.event_observed

    # Initialize variables to keep track of the best configuration
    best_p_value = 1.0
    best_group_config = (kmf_high_dls, kmf_low_dls)

    # Iterate through all possible configurations of moving patients between the groups
    for i in range(len(time_high_dls)):
        for j in range(len(time_low_dls)):
            # Create new configurations by moving patients
            new_time_high_dls = np.concatenate((time_high_dls[:i], time_low_dls[j:]))
            new_time_low_dls = time_high_dls[i:j]
            new_status_high_dls = np.concatenate((status_high_dls[:i], status_low_dls[j:]))
            new_status_low_dls = status_high_dls[i:j]

            # Initialize Kaplan-Meier estimator objects for the new configurations
            kmf_new_high_dls = KaplanMeierFitter()
            kmf_new_low_dls = KaplanMeierFitter()

            # Fit the Kaplan-Meier estimator to the data for the new configurations
            kmf_new_high_dls.fit(new_time_high_dls, event_observed=new_status_high_dls, label='High DLS')
            kmf_new_low_dls.fit(new_time_low_dls, event_observed=new_status_low_dls, label='Low DLS')

            # Calculate the p-value for the new configurations
            results = logrank_test(new_time_high_dls, new_time_low_dls, event_observed_A=new_status_high_dls, event_observed_B=new_status_low_dls)
            p_value = results.p_value

            # Update the best configuration if the p-value is smaller
            if p_value < best_p_value:
                best_p_value = p_value
                best_group_config = (kmf_new_high_dls, kmf_new_low_dls)

    return best_group_config



patient_info_list = load_and_predict(folder, results, modelpetct1)
patient_info_list_with_data = load_additional_data(csv, patient_info_list)

# # Printing the combined patient information list
# for patient_info in patient_info_list_with_data:
#     print(f"Patient ID: {patient_info['patient_folder']}, DLS Score: {patient_info['average_score']}, "
#           f"Time: {patient_info.get('Time', 'Not available')}, Status: {patient_info.get('Status', 'Not available')}")
# kaplan = kaplan_meier_analysis(patient_info_list)
# print(kaplan)

kmf_high_dls, kmf_low_dls = kaplan_meier_analysis(patient_info_list)
best_group_config = group_config(kmf_high_dls, kmf_low_dls)

# Plot the Kaplan-Meier survival curves for the best configuration
plt.figure(figsize=(10, 6))
best_group_config[0].plot(color='red')
best_group_config[1].plot(color='blue')
plt.title('Kaplan-Meier Survival Curve (Optimal Configuration)')
plt.xlabel('Time (months)')
plt.ylabel('Survival Probability')
plt.grid()
plt.legend(loc='lower left')
plt.show()


