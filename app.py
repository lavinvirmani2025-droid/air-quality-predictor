import os
import joblib
import requests

# Define the model filename
MODEL_FILENAME = 'air_quality_model.pkl'

# Check if the model file exists locally
if not os.path.exists(MODEL_FILENAME):
    print("Model not found locally. Downloading...")
    # Google Drive direct download link
    file_id = '14sCTd_pBAWek4u8GjRxvTgyfxHXXAB-3'
    url = f'https://drive.google.com/uc?export=download&id={file_id}'
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(MODEL_FILENAME, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Model downloaded successfully: {MODEL_FILENAME}")
    else:
        print("Failed to download the model.")
        exit(1)

# Load the model
model = joblib.load(MODEL_FILENAME)
