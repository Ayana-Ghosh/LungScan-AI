import tensorflow as tf
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')   # 🔥 add this before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def make_gradcam_heatmap(image_path, model, pred_index=None):
    # last_conv_layer_name = model.get_layer("conv5_block3_out").output
    # last_conv_layer_name = 'conv5_block3_out'
    last_conv_layer_name = 'conv5_block3_out'
    model = tf.keras.applications.ResNet50(weights='imagenet')

    # Load and preprocess image
    image = tf.keras.utils.load_img(image_path, target_size=(224, 224))  # Load image and resize
    img_arr = tf.keras.utils.img_to_array(image)  # Convert image to array
    img_bat = tf.expand_dims(img_arr, 0)  # Add batch dimension

    # Build Grad-CAM model
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Forward pass
    img_arr_expanded = np.expand_dims(img_arr, axis=0)  # Add batch dimension
    # Step 1: Predict and compute gradient
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_arr_expanded)
        predicted_class = tf.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    # Step 2: Compute gradients w.r.t. conv layer output
    grads = tape.gradient(loss, conv_outputs)
    # Step 3: Global average pooling over gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Step 4: Multiply each channel by the corresponding importance (pooled grad)
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Step 5: Normalize heatmap between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    return heatmap



def save_and_overlay_gradcam(img_path, heatmap, cam_path="static/gradcam/cam.jpg", alpha=0.4, img_size=224):
    from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import cv2
import numpy as np
import matplotlib.pyplot as plt

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

