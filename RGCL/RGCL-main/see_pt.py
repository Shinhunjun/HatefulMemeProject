import torch

# Load the .pt file
file_path = "/home/dhruv/Documents/Courses/EECE7205/Project/Code/RGCL-main_MyCopy/data/CLIP_Embedding/HarMeme (Original)/train_openai_clip-vit-large-patch14-336_HF.pt"
data = torch.load(file_path, weights_only=True)

# Print the keys in the file (if it's a dictionary)
if isinstance(data, dict):
    print("Keys in the .pt file:")
    for key in data.keys():
        print(key)
        # print(f'Example: {data[key][:5]}')
else:
    print("Data type in .pt file:", type(data))
