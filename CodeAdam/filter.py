import pickle
import shutil
import dicom2nifti
import re
import pandas as pd
import csv
import pydicom
import os

import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from sklearn.model_selection import KFold
from tqdm import tqdm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from lifelines import CoxPHFitter
import seaborn as sns
from lifelines import CoxPHFitter
from sklearn.model_selection import KFold
from collections import defaultdict

import warnings


columns_to_check = {'MORPHOLOGICAL_ApproximateVolume(IBSI:YEKZ)[mm3]'
    , 'MORPHOLOGICAL_SurfaceArea(IBSI:C0JK)[mm2]'
    , 'MORPHOLOGICAL_SurfaceToVolumeRatio(IBSI:2PR5)[mm]'
    , 'MORPHOLOGICAL_Compactness1(IBSI:SKGS)[]'
    , 'MORPHOLOGICAL_RadiusSphereNorm-MaxIntensityCoor-RoiCentroidCoor-Dist(IBSI:No)[]'
    , 'MORPHOLOGICAL_RadiusSphereNorm-MaxIntensityCoor-PerimeterCoor-3DSmallestDist(IBSI:No)[]'
    , 'MORPHOLOGICAL_Maximum3DDiameter(IBSI:L0JK)[mm]'
    , 'INTENSITY-BASED_MeanIntensity(IBSI:Q4LE)[]'
    , 'INTENSITY-BASED_IntensityVariance(IBSI:ECT3)[]'
    , 'INTENSITY-BASED_IntensitySkewness(IBSI:KE2A)[]'
    , 'INTENSITY-BASED_IntensityKurtosis(IBSI:IPH6)[]'
    , 'INTENSITY-BASED_MedianIntensity(IBSI:Y12H)[]'
    , 'INTENSITY-BASED_25thIntensityPercentile(IBSI:No)[]'
    , 'INTENSITY-BASED_50thIntensityPercentile(IBSI:Y12H)[]'
    , 'INTENSITY-BASED_75thIntensityPercentile(IBSI:No)[]'
    , 'INTENSITY-BASED_StandardDeviation(IBSI:No)[]'
    , 'INTENSITY-BASED_IntensityInterquartileRange(IBSI:SALO)[]'
    , 'INTENSITY-BASED_IntensityRange(IBSI:2OJQ)[]'
    , 'INTENSITY-BASED_IntensityBasedCoefficientOfVariation(IBSI:7TET)[]'
    , 'INTENSITY-HISTOGRAM_IntensityHistogramMean(IBSI:X6K6)[Intensity]'
    , 'INTENSITY-HISTOGRAM_IntensityHistogramVariance(IBSI:CH89)[Intensity]'
    , 'INTENSITY-HISTOGRAM_IntensityHistogramSkewness(IBSI:88K1)[Intensity]'
    , 'INTENSITY-HISTOGRAM_IntensityHistogramKurtosis(IBSI:C3I7)[Intensity]'
    , 'INTENSITY-HISTOGRAM_IntensityHistogramMedian(IBSI:WIFQ)[Intensity]'
    , 'INTENSITY-HISTOGRAM_IntensityHistogramMinimumGreyLevel(IBSI:1PR8)[Intensity]'
    , 'INTENSITY-HISTOGRAM_IntensityHistogram10thPercentile(IBSI:GPMT)[]'
    , 'INTENSITY-HISTOGRAM_IntensityHistogram25thPercentile(IBSI:No)[]'
    , 'INTENSITY-HISTOGRAM_IntensityHistogram50thPercentile(IBSI:No)[]'
    , 'INTENSITY-HISTOGRAM_IntensityHistogram75thPercentile(IBSI:No)[]'
    , 'INTENSITY-HISTOGRAM_IntensityHistogram90thPercentile(IBSI:OZ0C)[]'
    , 'INTENSITY-HISTOGRAM_IntensityHistogramStd(IBSI:No)[Intensity]'
    , 'INTENSITY-HISTOGRAM_IntensityHistogramMaximumGreyLevel(IBSI:3NCY)[Intensity]'
    , 'INTENSITY-HISTOGRAM_IntensityHistogramInterquartileRange(IBSI:WR0O)[Intensity]'
    , 'INTENSITY-HISTOGRAM_IntensityHistogramRange(IBSI:5Z3W)[Intensity]'
    , 'INTENSITY-HISTOGRAM_IntensityHistogramCoefficientOfVariation(IBSI:CWYJ)[Intensity]'
    , 'GLCM_JointMaximum(IBSI:GYBY)'
    , 'GLCM_JointAverage(IBSI:60VM)'
    , 'GLCM_JointVariance(IBSI:UR99)'
    , 'GLCM_JointVariance(IBSI:UR99)'
    , 'GLCM_JointEntropyLog2(IBSI:TU9B)'
    , 'GLCM_JointEntropyLog10(IBSI:No)'
    , 'GLCM_DifferenceAverage(IBSI:TF7R)'
    , 'GLCM_DifferenceVariance(IBSI:D3YU)'
    , 'GLCM_DifferenceEntropy(IBSI:NTRS)'
    , 'GLCM_SumAverage(IBSI:ZGXS)'
    , 'GLCM_SumVariance(IBSI:OEEB)'
    , 'GLCM_SumEntropy(IBSI:P6QZ)'
    , 'GLCM_AngularSecondMoment(IBSI:8ZQL)'
    , 'GLCM_Contrast(IBSI:ACUI)'
    , 'GLCM_Dissimilarity(IBSI:8S9J)'
    , 'GLCM_InverseDifference(IBSI:IB1Z)'
    , 'GLCM_NormalisedInverseDifference(IBSI:NDRX)'
    , 'GLCM_InverseDifferenceMoment(IBSI:WF0Z)'
    , 'GLCM_NormalisedInverseDifferenceMoment(IBSI:1QCO)'
    , 'GLCM_InverseVariance(IBSI:E8JP)'
    , 'GLCM_Correlation(IBSI:NI2N)'
    , 'GLCM_Autocorrelation(IBSI:QWB0)'
    , 'GLCM_ClusterTendency(IBSI:DG8W)'
    , 'GLCM_ClusterShade(IBSI:7NFM)'
    , 'GLCM_ClusterProminence(IBSI:AE86)'
    , 'GLRLM_ShortRunsEmphasis(IBSI:22OV)'
    , 'GLRLM_ShortRunsEmphasis(IBSI:22OV)'
    , 'GLRLM_LowGreyLevelRunEmphasis(IBSI:V3SW)'
    , 'GLRLM_HighGreyLevelRunEmphasis(IBSI:G3QZ)'
    , 'GLRLM_ShortRunLowGreyLevelEmphasis(IBSI:HTZT)'
    , 'GLRLM_ShortRunHighGreyLevelEmphasis(IBSI:GD3A)'
    , 'GLRLM_LongRunLowGreyLevelEmphasis(IBSI:IVPO)'
    , 'GLRLM_LongRunHighGreyLevelEmphasis(IBSI:3KUM)'
    , 'GLRLM_GreyLevelNonUniformity(IBSI:R5YN)'
    , 'GLRLM_RunLengthNonUniformity(IBSI:W92Y)'
    , 'GLRLM_RunPercentage(IBSI:9ZK5)'
    , 'NGTDM_Coarseness(IBSI:QCDE)'
    , 'NGTDM_Contrast(IBSI:65HE)'
    , 'NGTDM_Busyness(IBSI:NQ30)'
    , 'NGTDM_Complexity(IBSI:HDEZ)'
    , 'NGTDM_Strength(IBSI:1X9X)'
    , 'GLSZM_SmallZoneEmphasis(IBSI:5QRC)'
    , 'GLSZM_LargeZoneEmphasis(IBSI:48P8)'
    , 'GLSZM_LowGrayLevelZoneEmphasis(IBSI:XMSY)'
    , 'GLSZM_HighGrayLevelZoneEmphasis(IBSI:5GN9)'
    , 'GLSZM_SmallZoneLowGreyLevelEmphasis(IBSI:5RAI)'
    , 'GLSZM_SmallZoneHighGreyLevelEmphasis(IBSI:HW1V)'
    , 'GLSZM_LargeZoneLowGreyLevelEmphasis(IBSI:YH51)'
    , 'GLSZM_LargeZoneHighGreyLevelEmphasis(IBSI:J17V)'
    , 'GLSZM_GreyLevelNonUniformity(IBSI:JNSA)'
    , 'GLSZM_NormalisedGreyLevelNonUniformity(IBSI:Y1RO)'
    , 'GLSZM_ZoneSizeNonUniformity(IBSI:4JP3)'
    , 'GLSZM_NormalisedZoneSizeNonUniformity(IBSI:VB3A)'
    , 'GLSZM_ZonePercentage(IBSI:P30P)'
    , 'GLSZM_GreyLevelVariance(IBSI:BYLV)'
    , 'GLSZM_ZoneSizeVariance(IBSI:3NSA)'
    , 'GLSZM_ZoneSizeEntropy(IBSI:GU8N)'}

columns_to_check = list(columns_to_check)


def find_and_remove_folders_without_dcm_nii(root_folder):
    # Step 1: Collect directories with both DCM and NII files
    def contains_dcm_and_nii(dirpath):
        dcm_found = False
        nii_found = False
        for _, _, filenames in os.walk(dirpath):
            for filename in filenames:
                if filename.lower().endswith(".dcm"):
                    dcm_found = True
                elif filename.lower().endswith(".nii"):
                    nii_found = True
                if dcm_found and nii_found:
                    return True
        return False

    for dirpath, _, _ in os.walk(root_folder, topdown=False):
        if not contains_dcm_and_nii(dirpath):
            print(f"Removing directory and its parent folders: {dirpath}")
            try:
                shutil.rmtree(dirpath)
            except OSError as e:
                print(f"Error deleting directory {dirpath}: {str(e)}")


def rename_folders(root_folder):
    for patient_folder in os.listdir(root_folder):
        patient_folder_path = os.path.join(root_folder, patient_folder)
        if os.path.isdir(patient_folder_path):
            folders_with_dcm_nii = []

            # Step 1: Identify folders with both DCM and NII files for each patient
            for dirpath, dirnames, filenames in os.walk(patient_folder_path):
                dcm_found = any(f.lower().endswith(".dcm") for f in filenames)
                nii_found = any(f.lower().endswith(".nii") for f in filenames)

                if dcm_found and nii_found:
                    folders_with_dcm_nii.append(dirpath)

            # Step 2: Rename folders and create .txt files for each patient
            scan_counter = {}
            for folder_path in folders_with_dcm_nii:
                base_name = "scan"
                count = scan_counter.get(patient_folder_path, 0)
                if count > 0:
                    base_name += f"_{count}"
                scan_counter[patient_folder_path] = count + 1

                new_folder_path = os.path.join(patient_folder_path, base_name)

                try:
                    os.rename(folder_path, new_folder_path)

                    # Create a .txt file with the previous folder name for record-keeping
                    txt_filename = os.path.basename(folder_path) + ".txt"
                    txt_filepath = os.path.join(new_folder_path, txt_filename)

                    with open(txt_filepath, "w") as txt_file:
                        txt_file.write("Folder was renamed to: " + base_name)

                except FileExistsError as e:
                    print(f"Folder {new_folder_path} already exists. Skipping.")


def delete_empty_folders(root_folder):
    for dirpath, dirnames, filenames in os.walk(root_folder, topdown=False):
        # Check if the current directory is empty (contains no files or subdirectories)
        if not dirnames and not filenames:
            print(f"Deleting empty folder: {dirpath}")
            try:
                os.rmdir(dirpath)  # Delete the empty folder
            except OSError as e:
                print(f"Error deleting folder {dirpath}: {str(e)}")


def create_txt_folder(root_folder):
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(".txt"):
                txt_filepath = os.path.join(dirpath, filename)
                folder_name = os.path.splitext(filename)[0]  # Remove the file extension

                # Create a folder with the same name as the txt file (if it doesn't exist)
                folder_path = os.path.join(dirpath, folder_name)
                os.makedirs(folder_path, exist_ok=True)

                # Move the txt file into the folder
                new_txt_filepath = os.path.join(folder_path, filename)
                shutil.move(txt_filepath, new_txt_filepath)


def convert_dicom_series_to_nifti_recursive(root_folder):
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(".dcm"):
                dicom_files = [os.path.join(dirpath, f) for f in filenames if f.lower().endswith(".dcm")]

                try:
                    dicom_series = [pydicom.dcmread(dicom_file) for dicom_file in dicom_files]
                    nifti_image = dicom2nifti.dicom_series_to_nifti(dicom_series)
                except dicom2nifti.exceptions.ConversionError as e:
                    print(f"Error converting DICOM series in {dirpath}: {str(e)}")
                    continue

                # Save the NIfTI file at the same location as the DICOM series
                nifti_filepath = os.path.join(dirpath, "scan.nii.gz")
                dicom2nifti.write_nifti(nifti_image, nifti_filepath)





def list_folder_names_at_depth_2(root_dir):
    folder_names = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        depth = dirpath[len(root_dir):].count(os.sep)

        if depth == 2:
            for dirname in dirnames:
                folder_names.append(dirname)

    # Sort the folder names in ascending order
    folder_names.sort()

    return folder_names




def group_folders_by_similarity(root_dir, min_common_length=5):
    folder_groups = {}

    for dirpath, dirnames, filenames in os.walk(root_dir):
        depth = dirpath[len(root_dir):].count(os.sep)

        if depth == 2:
            for dirname in dirnames:
                lowercase_dirname = dirname.lower()

                # Check if the lowercase dirname contains a single word of min_common_length letters
                words = re.findall(r'\b\w{5,}\b', lowercase_dirname)

                if len(words) == 1:
                    common_key = words[0]

                    # Extract the patient ID from the path (assuming it's part of the directory structure)
                    patient_id = os.path.basename(os.path.dirname(dirpath))[-7:]  # Extract last 7 characters

                    if common_key not in folder_groups:
                        folder_groups[common_key] = []

                    folder_groups[common_key].append(patient_id)

    return folder_groups


def list_files_with_key(root_directory, key):
    file_list = []

    for foldername, subfolders, filenames in os.walk(root_directory):
        for filename in filenames:
            if key in filename:
                file_list.append(os.path.join(foldername, filename))

    file_list.sort()  # Sort the list of files in ascending order

    return file_list


def rename_files_with_key(root_directory, key, new_name):
    renamed_files = []  # To store the renamed file paths

    for foldername, subfolders, filenames in os.walk(root_directory):
        for filename in filenames:
            if key in filename:
                old_path = os.path.join(foldername, filename)
                new_filename = new_name
                new_path = os.path.join(foldername, new_filename)
                os.rename(old_path, new_path)
                renamed_files.append((old_path, new_path))

    return renamed_files


def copy_folders_with_subfolder_structure(root_directory, key, destination_directory):
    # Iterate through folders in the root directory
    for foldername, subfolders, filenames in os.walk(root_directory):
        for subfolder in subfolders:
            if key.lower() in subfolder.lower():
                # If the 'key' is found in a subfolder's name, copy the entire directory structure
                source_folder = os.path.join(foldername, subfolder)
                destination_folder = os.path.join(destination_directory, os.path.relpath(source_folder, root_directory))

                # Copy the directory structure, overwriting existing files
                for dirpath, dirnames, files in os.walk(source_folder):
                    relative_dirpath = os.path.relpath(dirpath, source_folder)
                    new_dirpath = os.path.join(destination_folder, relative_dirpath)

                    if not os.path.exists(new_dirpath):
                        os.makedirs(new_dirpath)

                    for file in files:
                        source_file = os.path.join(dirpath, file)
                        destination_file = os.path.join(new_dirpath, file)
                        shutil.copy2(source_file, destination_file)


def extract_dicom_info(dicom_folder):
    filter_type = "N/A"
    body_part = "N/A"
    convolution_kernel = "N/A"
    cine_rate = "N/A"
    contrast_bolus_agent = "N/A"

    for dicom_file in os.listdir(dicom_folder):
        dicom_file_path = os.path.join(dicom_folder, dicom_file)
        if os.path.isfile(dicom_file_path) and dicom_file_path.lower().endswith('.dcm'):
            try:
                dicom_data = pydicom.dcmread(dicom_file_path)
                filter_type = dicom_data.get("FilterType", "N/A")
                body_part = dicom_data.get("BodyPartExamined", "N/A")
                convolution_kernel = dicom_data.get("ConvolutionKernel", "N/A")
                cine_rate = dicom_data.get("CineRate", "N/A")
                contrast_bolus_agent = dicom_data.get("ContrastBolusAgent", "N/A")
                break  # Stop after finding information in the first DICOM file
            except Exception as e:
                pass  # Continue if there's an error with the DICOM file

    return filter_type, body_part, convolution_kernel, cine_rate, contrast_bolus_agent


def list_folders(root_directory, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            ['Folder', 'Subfolder', 'Depth3Folder', 'FilterType', 'BodyPart', 'ConvolutionKernel', 'CineRate',
             'ContrastBolusAgent'])

        for folder in os.listdir(root_directory):
            folder_path = os.path.join(root_directory, folder)

            if os.path.isdir(folder_path):
                for subfolder in os.listdir(folder_path):
                    subfolder_path = os.path.join(folder_path, subfolder)

                    if os.path.isdir(subfolder_path):
                        for depth3_folder in os.listdir(subfolder_path):
                            depth3_path = os.path.join(subfolder_path, depth3_folder)
                            if os.path.isdir(depth3_path):
                                filter_type, body_part, convolution_kernel, cine_rate, contrast_bolus_agent = extract_dicom_info(
                                    depth3_path)
                                csv_writer.writerow(
                                    [folder, subfolder, depth3_folder, filter_type, body_part, convolution_kernel,
                                     cine_rate, contrast_bolus_agent])


def generate_texture_config(root_directory, output_file):
    with open(output_file, 'w') as config_file:
        session_number = 0
        for patient_folder in os.listdir(root_directory):
            if os.path.isdir(os.path.join(root_directory, patient_folder)):
                scan_file = None
                roi_file = None

                patient_path = os.path.join(root_directory, patient_folder)
                for dirpath, _, filenames in os.walk(patient_path):
                    for filename in filenames:
                        if filename.endswith("scan.nii.gz"):
                            scan_file = os.path.join(dirpath, filename)
                        elif filename.endswith(".nii"):
                            roi_file = os.path.join(dirpath, filename)

                if scan_file and roi_file:
                    config_file.write(f"LIFEx.texture.Session{session_number}.Img0={scan_file}\n")
                    config_file.write(f"LIFEx.texture.Session{session_number}.Roi0={roi_file}\n")
                    session_number += 1

                    # Add two empty lines
                    config_file.write("\n\n")


def compare_common_folders(folder1_path, folder2_path, common_folder_check_path):
    # Get a list of folder names in the first directory
    folder1_names = set(os.listdir(folder1_path))

    # Get a list of folder names in the second directory
    folder2_names = set(os.listdir(folder2_path))

    # Find the common folder names between the first and second directories
    common_folders = folder1_names.intersection(folder2_names)

    # Get a list of folder names in the third directory
    folder3_names = set(os.listdir(common_folder_check_path))

    # Find the common folder names between the common folders and the third directory
    common_folders_in_third = common_folders.intersection(folder3_names)

    # Find folders that are in folder1 and folder2 but not in the third folder
    missing_folders = common_folders - common_folders_in_third

    if missing_folders:
        print("Folders present in both folder1 and folder2 but not in the third folder:")
        for folder_name in missing_folders:
            print(folder_name)
    else:
        print("Everything is OK, all folders present.")







def generate_texture_config_v3(root_directory, output_file):
    with open(output_file, 'w') as config_file:
        patient_number = 0

        for patient_folder in os.listdir(root_directory):
            if os.path.isdir(os.path.join(root_directory, patient_folder)):
                scan_file = None
                roi_file = None

                patient_path = os.path.join(root_directory, patient_folder)
                for dirpath, _, filenames in os.walk(patient_path):
                    for filename in filenames:
                        if filename.endswith("scan.nii.gz"):
                            scan_file = os.path.join(dirpath, filename)
                        elif filename.endswith(".nii"):
                            roi_file = os.path.join(dirpath, filename)

                if scan_file and roi_file:
                    config_file.write(f'LIFEx.Patient{patient_number}.Series0={scan_file}\n')
                    config_file.write(f'LIFEx.Patient{patient_number}.Roi0={roi_file}\n')
                    config_file.write(f'LIFEx.Patient{patient_number}.Roi0.Operation0=Ring,3|Save nii\n')
                    config_file.write(
                        f'LIFEx.Patient{patient_number}.Roi0.Operation0.Output.Directory={os.path.dirname(scan_file)}\n')

                    patient_number += 1

                # Add two empty lines to separate patients
                config_file.write("\n\n")



warnings.filterwarnings("ignore")

binarized_values_df = pd.read_csv('/home/lito/Downloads/binarized_values_both.csv')

binarized_values_df['Status'] = (binarized_values_df['Status'] == 1).astype(int)
print(binarized_values_df['Status'])

X = binarized_values_df[columns_to_check]
y = binarized_values_df[['Status', 'Time']]


import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.model_selection import KFold
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

def backward_cox_regression(X, y, threshold_out=0.05, min_features=1, verbose=False):
    included = list(X.columns)

    while len(included) > min_features:
        changed = False

        # Fit a Cox model using the current set of included features
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(pd.concat([X[included], pd.DataFrame(y, columns=['Status', 'Time'])], axis=1),
                duration_col='Time', event_col='Status')

        pvalues = cph.summary['p']

        worst_pval = pvalues.max()

        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()

            included.remove(worst_feature)

            # if verbose:
            #     print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))

        if not changed:
            break

    # Calculate C-index for the final model
    c_index_final = cph.concordance_index_
    print(f"\nC-Index for the Final Model: {c_index_final}")

    # Filter out features with p-values above the threshold
    selected_features = [feature for feature in included if cph.summary.loc[feature, 'p'] < threshold_out]

    # If no features are selected, include the feature with the lowest p-value
    if not selected_features:
        best_feature = pvalues.idxmin()
        selected_features.append(best_feature)

    # Ensure at least the specified number of features are selected
    while len(selected_features) < min_features:
        for feature in included:
            if feature not in selected_features:
                selected_features.append(feature)
                break

    print("\nSelected Features:")
    for feature in selected_features:
        p_value = cph.summary.loc[feature, 'p']
        print(f"{feature:30} with p-value {p_value:.6f}")

    return selected_features, c_index_final




def forward_cox_regression(X, y, threshold_in=0.05, verbose=False):
    included = []
    cph = None  # Initialize cph outside the loop

    while True:
        changed = False
        best_pval = threshold_in
        best_feature = None

        for feature in X.columns:
            if feature not in included:
                candidate_features = included + [feature]
                new_cph = CoxPHFitter(penalizer=0.01)
                new_cph.fit(pd.concat([X[candidate_features], pd.DataFrame(y, columns=['Status', 'Time'])], axis=1),
                            duration_col='Time', event_col='Status')

                # Check if the feature is in new_cph.summary before accessing it
                if feature in new_cph.summary.index:
                    p_value = new_cph.summary.loc[feature, 'p']

                    if p_value < best_pval:
                        best_pval = p_value
                        best_feature = feature
                        cph = new_cph  # Update cph if a new best feature is found

        if best_feature:
            changed = True
            included.append(best_feature)

            if verbose:
                print('Add {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # Print available features in new_cph.summary
        print("Available features in new_cph.summary:", new_cph.summary.index)

        # Break the loop when no more features can be added
        if not changed:
            break

    # Calculate C-index for the final model
    final_features = X[included]
    final_cph = CoxPHFitter(penalizer=0.1)
    final_cph.fit(pd.concat([final_features, pd.DataFrame(y, columns=['Status', 'Time'])], axis=1),
                  duration_col='Time', event_col='Status')
    c_index_final = final_cph.concordance_index_

    return included, c_index_final


def calculate_aic(X, y, selected_features):
    X_selected = X[selected_features]
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(pd.concat([X_selected, pd.DataFrame(y, columns=['Status', 'Time'])], axis=1), duration_col='Time', event_col='Status')
    aic = cph.AIC_partial_
    return aic


def backward_stepwise_aic(X, y):
    included = list(X.columns)
    best_aic = np.inf
    selected_features = []

    while len(included) > 0:
        aic_values = []

        for feature in included:
            candidate_features = included.copy()
            candidate_features.remove(feature)
            aic = calculate_aic(X, y, candidate_features)
            aic_values.append((feature, aic))

        best_candidate, best_candidate_aic = min(aic_values, key=lambda x: x[1])

        if best_candidate_aic < best_aic:
            best_aic = best_candidate_aic
            selected_features.append(best_candidate)
            included.remove(best_candidate)
        else:
            break

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(pd.concat([X[selected_features], pd.DataFrame(y, columns=['Status', 'Time'])], axis=1), duration_col='Time', event_col='Status')
    c_index = cph.concordance_index_

    return selected_features, c_index


def create_fixed_folds(X, n_splits=5, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(kf.split(X))
    return folds

def cross_validate_backward_selection(X, y, folds, threshold_out=0.05, n_iterations=10, verbose=True):
    c_index_list = []
    selected_features_list = []
    feature_count_dict = defaultdict(int)

    for iteration in tqdm(range(n_iterations), desc="Iterations", total=n_iterations):
        for fold, (train_index, test_index) in tqdm(enumerate(folds), desc="Folds", total=len(folds), leave=False):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            selected_features, c_index = backward_cox_regression(X_train, y_train,threshold_out, min_features = 1)

            feature_count_dict[len(selected_features)] += 1

            c_index_list.append(c_index)
            selected_features_list.append(selected_features)

            if verbose:
                print(f"Iteration {iteration + 1}, Fold {fold + 1} - Selected Features: {selected_features}")

    return c_index_list, selected_features_list, feature_count_dict

def cross_validate_forward_selection(X, y, folds, threshold_in=0.05, n_iterations=10, verbose=False):
    c_index_list = []
    selected_features_list = []
    feature_count_dict = defaultdict(int)

    for iteration in tqdm(range(n_iterations), desc="Iterations", total=n_iterations):
        for fold, (train_index, test_index) in tqdm(enumerate(folds), desc="Folds", total=len(folds), leave=False):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            selected_features, c_index = forward_cox_regression(X_train, y_train, threshold_in, verbose=True)

            feature_count_dict[len(selected_features)] += 1

            c_index_list.append(c_index)
            selected_features_list.append(selected_features)

            if verbose:
                print(f"Iteration {iteration + 1}, Fold {fold + 1} - Selected Features: {selected_features}")

    return c_index_list, selected_features_list, feature_count_dict



def cross_validate_stepaic(X, y, folds, n_iterations=10, verbose=False):
    c_index_list = []
    selected_features_list = []
    feature_count_dict = defaultdict(int)

    for iteration in tqdm(range(n_iterations), desc="Iterations", total=n_iterations):
        for fold, (train_index, test_index) in tqdm(enumerate(folds), desc="Folds", total=len(folds), leave=False):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            selected_features, c_index = backward_stepwise_aic(X_train, y_train)

            if verbose:
                print(f"\nSelected Features (Iteration {iteration + 1}, Fold {fold + 1}):")
                for feature in selected_features:
                    print(feature)

            # Update feature count dictionary
            feature_count_dict[len(selected_features)] += 1

            selected_features_list.append(selected_features)
            c_index_list.append(c_index)

    return c_index_list, selected_features_list, feature_count_dict



def calculate_vif(data_frame):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = data_frame.columns
    vif_data["VIF"] = [variance_inflation_factor(data_frame.values, i) for i in range(data_frame.shape[1])]

    # Get feature names that are not 'inf'
    valid_features = set(vif_data.loc[vif_data["VIF"] != float('inf'), "Variable"])

    return vif_data, valid_features

def cross_validate_vif(X, y, folds, n_iterations=10):
    c_index_list = []
    selected_features_list = []
    feature_count_dict = defaultdict(int)

    for iteration in tqdm(range(n_iterations), desc="Iterations", total=n_iterations):
        for fold, (train_index, test_index) in tqdm(enumerate(folds), desc="Folds", total=len(folds), leave=False):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Calculate VIF and exclude features with 'inf' VIF
            vif_result, valid_features = calculate_vif(X_train)

            # Fit Cox model with the remaining variables
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(pd.concat([X_train[list(valid_features)], pd.DataFrame(y_train, columns=['Status', 'Time'])], axis=1),
                    duration_col='Time', event_col='Status')

            selected_features = list(valid_features)

            # Update feature count dictionary
            feature_count_dict[len(selected_features)] += 1

            selected_features_list.append(selected_features)

            c_index = cph.concordance_index_
            c_index_list.append(c_index)

    return c_index_list, selected_features_list, feature_count_dict



#################################PLOTS###########################


def plot_feature_selection_frequency(selected_features_list, method_name):
    all_selected_features = [feature for fold_features in selected_features_list for feature in fold_features]

    feature_counts = pd.Series(all_selected_features).value_counts().reset_index()
    feature_counts.columns = ['Feature', 'Frequency']

    # Sort features by frequency
    feature_counts = feature_counts.sort_values(by='Frequency', ascending=False)

    plt.figure(figsize=(100, 300))
    sns.barplot(x='Frequency', y='Feature', data=feature_counts, palette='viridis')
    plt.title(f'Frequency of Feature Selection - {method_name}')
    plt.xlabel('Frequency')
    plt.ylabel('Feature')
    plt.show()


def plot_c_index_histogram(c_index_list, method_name):
    mean_c_index = np.mean(c_index_list)
    std_c_index = np.std(c_index_list)

    # Set up the plot
    plt.figure(figsize=(10, 6))
    # Plot histogram with KDE
    sns.histplot(c_index_list, kde=True, color='skyblue', element='step')

    # Add a line representing the evolution of the C-Index
    plt.axvline(mean_c_index, color='red', linestyle='dashed', linewidth=2, label='Mean C-Index')

    # Customize plot labels and title
    plt.title(f'Distribution of C-Index - {method_name}\nMean C-Index: {mean_c_index:.4f}, Std: {std_c_index:.4f}', fontsize=14)
    plt.xlabel('Concordance Index (C-Index)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

    # Add legend
    plt.legend()

    # Display the plot
    plt.show()


def plot_feature_count(feature_count_dict, method_name):
    feature_counts = sorted(feature_count_dict.items())

    plt.figure(figsize=(10, 6))
    sns.barplot(x=[str(count) for count, _ in feature_counts], y=[count for _, count in feature_counts],
                color='skyblue')
    plt.title(f'Number of Models vs Number of Selected Features - {method_name}')
    plt.xlabel('Number of Selected Features')
    plt.ylabel('Number of Models')
    plt.show()


def plot_cindex_boxplot(c_index_lists, method_names):
    plt.figure(figsize=(15, 8))

    dfs = []

    for method_name, c_index_list in zip(method_names, c_index_lists):
        df = pd.DataFrame({'Method': [method_name] * len(c_index_list),
                           'C-Index': c_index_list})
        dfs.append(df)

    concatenated_df = pd.concat(dfs)

    sns.boxplot(x='Method', y='C-Index', data=concatenated_df, showfliers=False, width=0.5)
    sns.stripplot(x='Method', y='C-Index', data=concatenated_df, size=8, color='black', jitter=True, alpha=0.7)

    plt.title('Distribution of C-Index for Different Feature Selection Methods')
    plt.xlabel('Feature Selection Method')
    plt.ylabel('C-Index')
    plt.show()


def plot_cindex_vs_features(c_index_lists, method_names):
    plt.figure(figsize=(10, 6))

    for c_index_list, method_name in zip(c_index_lists, method_names):
        # Create lists to store data for plotting
        x_values = []
        y_values = []

        # Iterate over models and populate the lists
        for model_id, cindex in enumerate(c_index_list):
            x_values.append(model_id + 1)
            y_values.append(cindex)

        # Plotting
        plt.scatter(x_values, y_values, marker='o', label=method_name)

    plt.title('C-Index vs Number of Features for Different Feature Selection Methods')
    plt.xlabel('Number of Features')
    plt.ylabel('C-Index')
    plt.legend()
    plt.grid(True)
    plt.show()




folds = create_fixed_folds(X, n_splits=5, random_state=42)

iterations = 10

# Forward Selection
c_index_list_forward, selected_features_list_forward, feature_count_dict_forward = cross_validate_forward_selection(X, y, folds, threshold_in=0.05, n_iterations=iterations, verbose=True)

# VIF-based Selection
c_index_list_vif, selected_features_list_vif, feature_count_dict_vif = cross_validate_vif(X, y, folds, n_iterations=iterations)

# StepAIC
c_index_list_stepaic, selected_features_list_stepaic, feature_count_dict_stepaic = cross_validate_stepaic(X, y, folds, n_iterations=iterations, verbose=True)

# # Backward Selection
c_index_list_backward, selected_features_list_backward, feature_count_dict_backward = cross_validate_backward_selection(X, y, folds, threshold_out=0.05, n_iterations=iterations, verbose=True)

with open('/home/lito/Desktop/grp_a_lists_folds_both.pkl', 'wb') as file:
    data_to_save = {
        'forward': (c_index_list_forward, selected_features_list_forward, feature_count_dict_forward),
        'vif': (c_index_list_vif, selected_features_list_vif, feature_count_dict_vif),
        'stepaic': (c_index_list_stepaic, selected_features_list_stepaic, feature_count_dict_stepaic),
        'backward': (c_index_list_backward, selected_features_list_backward, feature_count_dict_backward)
    }
    pickle.dump(data_to_save, file)


# plot_feature_selection_frequency(selected_features_list_backward, 'Backward Selection')
# # plot_feature_selection_frequency(selected_features_list_forward, 'Forward Selection')
# plot_feature_selection_frequency(selected_features_list_vif, 'VIF Selection')
# plot_feature_selection_frequency(selected_features_list_stepaic, 'Backward Stepwise AIC')
#
# plot_c_index_histogram(c_index_list_backward, 'Backward Selection')
# plot_c_index_histogram(c_index_list_forward, 'Forward Selection')
# plot_c_index_histogram(c_index_list_vif, 'VIF Selection')
# plot_c_index_histogram(c_index_list_stepaic, 'Backward Stepwise AIC')
#
# plot_feature_count(feature_count_dict_backward, 'Backward Selection')
# plot_feature_count(feature_count_dict_forward, 'Forward Selection')
# plot_feature_count(feature_count_dict_vif, 'VIF-based Selection')
# plot_feature_count(feature_count_dict_stepaic, 'Backward Stepwise AIC')
#
# plot_cindex_boxplot([c_index_list_backward, c_index_list_forward, c_index_list_vif, c_index_list_stepaic],
#                     ['Backward Stepwise', 'Forward Stepwise', 'VIF', 'Backward Stepwise AIC'])
#
# plot_cindex_vs_features(c_index_list_backward, feature_count_dict_backward, 'Backward Selection')
# plot_cindex_vs_features(c_index_list_forward, feature_count_dict_forward, 'Forward Selection')
# plot_cindex_vs_features(c_index_list_vif, feature_count_dict_vif, 'VIF Selection')
# plot_cindex_vs_features(c_index_list_stepaic, feature_count_dict_stepaic, 'Stepwise AIC')
