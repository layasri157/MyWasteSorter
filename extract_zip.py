import zipfile

zip_path = r"C:\Users\DELL\Downloads\archive (1).zip"
extract_to = r"C:\Users\DELL\Projects\waste_dataset"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("Extraction complete.")
