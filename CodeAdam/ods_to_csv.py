import pandas as pd

def ods_to_csv(ods_file_path, csv_file_path):
    df = pd.read_excel(ods_file_path, engine="odf")
    df.to_csv(csv_file_path, index=False)

# Example usage:
ods_file_path = "/media/adamdiakite/LaCie/CAPRICORN_OS.ods"
csv_file_path = "/media/adamdiakite/LaCie/CT-TEP_ICI/Results/ICI.csv"
ods_to_csv(ods_file_path, csv_file_path)