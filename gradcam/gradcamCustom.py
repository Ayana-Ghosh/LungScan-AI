import tensorflow as tf
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')   # 🔥 Important for server-based or headless environments
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def make_gradcam_heatmap(image_path, model, pred_index=None):
    # --- Specify the last convolutional layer manually ---
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break

    if last_conv_layer_name is None:
        raise ValueError("No convolutional layer found in the model.")

    # --- Load and preprocess image ---
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # 🔥 Normalization for custom CNN

    # --- Build Grad-CAM model ---
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # --- Forward pass ---
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0]) if pred_index is None else pred_index
        class_channel = predictions[:, pred_index]

    # --- Compute gradients ---
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # --- Multiply pooled grads with feature maps ---
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # --- Normalize heatmap ---
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    return heatmap

def save_and_overlay_gradcam(img_path, heatmap, cam_path="static/gradcam/cam.jpg", alpha=0.4, img_size=224):
    # --- Load original image ---
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Resize and apply heatmap ---
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    # --- Superimpose heatmap ---
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)

    # --- Plot with colorbar ---
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(superimposed_img)
    ax.set_title("Grad-CAM with Color Intensity Scale", fontsize=14)
    ax.axis('off')

    # --- Add colorbar ---
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(cmap='jet', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Importance/Activation Level', rotation=270, labelpad=15)

    # --- Save figure ---
    plt.tight_layout()
    plt.savefig(cam_path, bbox_inches='tight', pad_inches=0.2)
    plt.close()

    return cam_path
