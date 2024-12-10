import os
import json
import easyocr

reader = easyocr.Reader(['en'])  # Add languages as needed

# Folder containing the images
folder_path = "/home/dhruv/Documents/Courses/EECE7205/Project/Code/RGCL-main_MyCopy/data/image/HarMeme/Test"

# JSONL file to save the output
output_file = "/home/dhruv/Documents/Courses/EECE7205/Project/Code/RGCL-main_MyCopy/data/gt/HarMeme/test_new.jsonl"

with open(output_file, "w") as jsonl_file:
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        if os.path.isfile(image_path) and filename.lower().endswith((".png")):
            results = reader.readtext(image_path)
            extracted_text = "\n".join([text for (_, text, _) in results])
            file_id = os.path.splitext(filename)[0]
            data = {"id": file_id, "image": filename, "labels": [], "text": extracted_text}
            jsonl_file.write(json.dumps(data) + "\n")

print(f"JSONL file 'test_new.jsonl' created successfully!")

