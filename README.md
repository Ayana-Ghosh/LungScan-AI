# LungScan-AI

A web application for **lung cancer detection** using advanced deep learning architectures and Grad-CAM visualizations.  
Users can upload a chest scan image, select a model, and get predictions along with Grad-CAM heatmaps highlighting key regions.

---

## Features

- Predicts **lung cancer risk** from uploaded chest images.
- Multiple pretrained deep learning models:
  - VGG16
  - ResNet50
  - Inception
  - Xception
  - Custom CNN
- Grad-CAM visualizations (local execution only)
- Clean, user-friendly web interface
- Flask-based backend

---

## Demo

- Live demo (predictions only, Grad-CAM disabled due to free hosting memory limits):  
[https://lungscan-ai.onrender.com](https://lungscan-ai.onrender.com)

---

## Deployment Note (Grad-CAM)

The Grad-CAM visualization feature is fully implemented and works correctly in local execution.

However, for the live deployment on **Render (free tier)**, Grad-CAM has been temporarily disabled due to strict memory limitations of the free hosting environment. Deep learning visualization (Grad-CAM + Matplotlib) causes high memory usage, which exceeds the available resources and leads to server restarts (HTTP 502).

This is an **infrastructure constraint**, not a limitation of the model or implementation.

All prediction functionalities work correctly in the deployed version.

---

## Installation
git clone https://github.com/Ayana-Ghosh/LungScan-AI.git

## Create a virtual environment and activate it:

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / Mac
source venv/bin/activate

## Install dependencies:
pip install -r requirements.txt


## Run the application locally:
python app.py

## Usage

Open the web app in your browser: http://127.0.0.1:5000/
Upload a chest scan image (must be in the same format as the dataset used for training, e.g., .png, .jpg).
Select a deep learning model.
Click Predict to get the lung cancer prediction.
If running locally, Grad-CAM heatmaps will show highlighted regions of interest.

## Limitations

1. The application cannot diagnose all types of images. Uploaded images must be similar in type to the dataset used for training (e.g., chest scans in the correct format).
2. The model does not guarantee medical diagnosis and should not be used as a replacement for professional medical advice.
3. Grad-CAM visualizations are disabled in the free deployment due to memory constraints. Full functionality is available only in local execution.

## Future Scope

1. Expand the dataset to include more diverse imaging sources and formats.
2. Integrate additional deep learning architectures for improved accuracy.
3. Deploy Grad-CAM visualization on a paid cloud server with higher memory for live use.
4. Add automatic preprocessing and validation for uploaded images to handle different formats and resolutions.
5. Explore integration with other imaging modalities (e.g., CT scans) for more comprehensive cancer detection.

## Contributing

Contributions are welcome! If you want to improve the application, fix bugs, or add new features. We appreciate any contributions to make LungScan-AI more robust and user-friendly.
