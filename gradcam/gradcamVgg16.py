import tensorflow as tf
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')   # 🔥 add this before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def make_gradcam_heatmap(image, model, pred_index=None):
    # Set the last convolutional layer name for VGG16
    last_conv_layer_name = 'block5_conv3'
    model = tf.keras.applications.VGG16(weights='imagenet')

    img = tf.keras.preprocessing.image.load_img(image, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)

    # Create a model that maps input image to activations of the last conv layer + model output
    grad_model = tf.keras.models.Model(
    [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # --- Forward pass ---
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0]) if pred_index is None else pred_index
        class_channel = predictions[:, pred_index]

    # --- Get gradients of predicted class with respect to last conv layer output ---
    grads = tape.gradient(class_channel, conv_outputs)

    # --- Global average pooling of gradients ---
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # --- Multiply pooled grads with feature maps ---
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # --- Normalize the heatmap between 0 and 1 ---
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    return heatmap  # Return the heatmap

def save_and_overlay_gradcam(img_path, heatmap, cam_path="static/gradcam/cam.jpg", alpha=0.4, img_size=224):
        # --- Step 1: Load original image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Step 2: Resize and apply heatmap
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    # --- Step 3: Superimpose heatmap on original image
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)

    # --- Step 4: Plot with colorbar
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(superimposed_img)
    ax.set_title("Grad-CAM with Color Intensity Scale", fontsize=14)
    ax.axis('off')

    # --- Step 5: Add colorbar linked to the ScalarMappable
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(cmap='jet', norm=norm)
    sm.set_array([])  # Dummy array just to enable colorbar
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Importance/Activation Level', rotation=270, labelpad=15)

    # --- Step 6: Save figure
    plt.tight_layout()
    plt.savefig(cam_path, bbox_inches='tight', pad_inches=0.2)
    plt.close()

    return cam_path
