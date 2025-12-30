from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from utils import load_pretrained_model, preprocess_image, predict_with_model, make_gradcam_heatmap, save_and_overlay_gradcam, is_valid_image
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template("error.html", error_message="No image file uploaded.")

    image_file = request.files['image']
    model_choice = request.form['model']

    if image_file.filename == '':
        return render_template("error.html", error_message="No selected file.")

    if image_file and allowed_file(image_file.filename):
        filename = secure_filename(image_file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        image_file.save(image_path)

        print("Image saved to:", image_path)

        if not is_valid_image(image_path):
            os.remove(image_path)
            return render_template("error.html", error_message="Invalid image. Please upload a valid JPG/PNG medical image.")

        try:
            model = load_pretrained_model(model_choice.lower())
        except ValueError as e:
            return render_template("error.html", error_message=str(e))

        image = preprocess_image(image_path)
        predicted_label, confidence = predict_with_model(model, image)
        result_text = f"{predicted_label} ({confidence:.2f}%)"

        # ---------------- Grad-CAM code commented out for Render deployment ----------------
        """
        heatmap = make_gradcam_heatmap(model, image_path, model_choice.lower())
        gradcam_path = os.path.join('static/gradcam', 'gradcam_' + unique_filename)
        save_and_overlay_gradcam(image_path, heatmap, gradcam_path, model_choice.lower())
        """
        # ---------------------------------------------------------------------------------

        return render_template("result.html",
                               image_file=unique_filename,
                               model=model_choice.upper(),
                               result=result_text,
                               gradcam_file=None)  # No Grad-CAM file for Render deployment
    else:
        return render_template("error.html", error_message="Unsupported file type. Please upload PNG, JPG or JPEG images only.")

if __name__ == '__main__':
    # Use Render's PORT environment variable, default to 5000 for local testing
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
