from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import tensorflow as tf
import cv2
import os
from PIL import Image
import imghdr

# Define your class labels
CATEGORIES = ['lung_adenocarcinomas', 'lung_normal', 'lung_squamous_cell_carcinomas']
IMG_SIZE = 224

def load_pretrained_model(model_name="resnet"):
    if model_name == "vgg16":
        return load_model('models/model_vgg16.h5')
    elif model_name == "resnet":
        return load_model('models/model_resnet.h5')
    elif model_name == "inception":
        return load_model('models/model_inception.h5')
    elif model_name == "xception":
        return load_model('models/model_xception.h5')
    else:
        return load_model('models/model_ccnn.h5')

def preprocess_image(img_path, target_size=(IMG_SIZE, IMG_SIZE)):
    """Preprocess the image for prediction (no dominant color distortion)."""
    image = load_img(img_path, target_size=target_size)
    img_array = img_to_array(image)
    img_array = preprocess_input(img_array)  # Important for pretrained models
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_with_model(model, image):
    """Make prediction and return label + confidence."""
    prediction = model.predict(image)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = CATEGORIES[predicted_class_index]
    confidence = np.max(prediction) * 100
    return predicted_class_label, confidence

def is_valid_image(file_path):
    """Check if image is a valid JPG/PNG."""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return imghdr.what(file_path) in ['jpeg', 'png', 'jpg']
    except Exception:
        return False

# Grad-CAM modules
from gradcam import gradcamVgg16, gradcamResnet, gradcamInception, gradcamXception, gradcamCustom

def make_gradcam_heatmap(model, image, model_name, pred_index=None):
    if model_name == "vgg16":
        return gradcamVgg16.make_gradcam_heatmap(image, model)
    elif model_name == "resnet":
        return gradcamResnet.make_gradcam_heatmap(image, model)
    elif model_name == "inception":
        return gradcamInception.make_gradcam_heatmap(image, model)
    elif model_name == "xception":
        return gradcamXception.make_gradcam_heatmap(image, model)
    else:
        return gradcamCustom.make_gradcam_heatmap(image, model)

def save_and_overlay_gradcam(img_path, heatmap, cam_path, model_name):
    if model_name == "vgg16":
        return gradcamVgg16.save_and_overlay_gradcam(img_path, heatmap, cam_path, model_name)
    elif model_name == "resnet":
        return gradcamResnet.save_and_overlay_gradcam(img_path, heatmap, cam_path, model_name)
    elif model_name == "inception":
        return gradcamInception.save_and_overlay_gradcam(img_path, heatmap, cam_path, model_name)
    elif model_name == "xception":
        return gradcamXception.save_and_overlay_gradcam(img_path, heatmap, cam_path, model_name)
    else:
        return gradcamCustom.save_and_overlay_gradcam(img_path, heatmap, cam_path, model_name)
