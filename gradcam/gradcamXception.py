import tensorflow as tf
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # 🔥 important
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def make_gradcam_heatmap(image_path, model=None, pred_index=None):
    # Set last conv layer name for Xception
    last_conv_layer_name = 'block14_sepconv2_act'
    model = tf.keras.applications.Xception(weights='imagenet')

    # Load and preprocess image
    image = tf.keras.utils.load_img(image_path, target_size=(299, 299))  # Xception expects 299x299
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.xception.preprocess_input(img_array)

    # Build Grad-CAM model
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predicted_class = tf.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    return heatmap

def save_and_overlay_gradcam(img_path, heatmap, cam_path="static/gradcam/cam_xception.jpg", alpha=0.4, img_size=299):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(superimposed_img)
    ax.set_title("Grad-CAM with Color Intensity Scale", fontsize=14)
    ax.axis('off')

    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(cmap='jet', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Importance/Activation Level', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(cam_path, bbox_inches='tight', pad_inches=0.2)
    plt.close()

    return cam_path
