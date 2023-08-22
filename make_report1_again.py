import os
import pandas as pd

folder_path = "./테스트2/제출파일1_최종"

csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    df = pd.read_csv(file_path, encoding='cp949')
    df.replace("Null", pd.NA, inplace=True)
    new_file_path = os.path.join(folder_path, f"new_{csv_file}")
    df.to_csv(new_file_path, index=False)
    
    print(f"Processed: {csv_file} -> {new_file_path}")
