import os
from flask import Flask, render_template, request, jsonify, send_file
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import tempfile
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Ensure the directory exists for saving temporary images
static_folder = 'static'
if not os.path.exists(static_folder):
    os.makedirs(static_folder)

# Load the model
model = load_model('model/paddy_model.keras')

# Manually define class labels
class_labels = ['bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight', 
                'blast', 'brown_spot', 'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro']

# Function for SIFT feature visualization
def show_sift_features(img_pil):
    img_cv = np.array(img_pil.convert('RGB'))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, _ = sift.detectAndCompute(gray, None)
    img_sift = cv2.drawKeypoints(img_cv, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Convert the SIFT image from BGR to RGB
    img_sift_rgb = cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB)
    
    return img_sift_rgb


# Function for LoG (Laplacian of Gaussian) feature extraction
def show_log_features(img_pil, sigma=1.0):
    img_cv = np.array(img_pil.convert('RGB'))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    log_img = cv2.GaussianBlur(gray, (0, 0), sigma)
    log_img = cv2.Laplacian(log_img, cv2.CV_64F)
    log_img = cv2.convertScaleAbs(log_img)
    return log_img

# Route to serve the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Open and preprocess the image
        img = Image.open(file.stream).convert('RGB')
        img_resized = img.resize((224, 224))  # Adjust size as per your model's input requirement
        img_array = np.array(img_resized) / 255.0
        img_array_expanded = np.expand_dims(img_array, axis=0)
        
        # Predict with the model
        prediction = model.predict(img_array_expanded)
        predicted_idx = np.argmax(prediction)
        confidence = prediction[0][predicted_idx] * 100
        predicted_label = class_labels[predicted_idx]
        
        # Visualize SIFT features
        sift_img = show_sift_features(img)
        # Visualize LoG features
        log_img = show_log_features(img)

        # Save the SIFT and LoG images in the static folder
        sift_img_path = os.path.join(static_folder, 'sift_img.png')
        log_img_path = os.path.join(static_folder, 'log_img.png')
        
        plt.imsave(sift_img_path, cv2.cvtColor(sift_img, cv2.COLOR_BGR2RGB))
        plt.imsave(log_img_path, log_img, cmap='gray')

        # Save the uploaded image in the static folder
        uploaded_img_path = os.path.join(static_folder, 'uploaded_image.png')
        img.save(uploaded_img_path)

        # Return results with images
        return render_template('index.html', prediction=predicted_label, confidence=confidence,
                               sift_image='/static/sift_img.png', log_image='/static/log_img.png', uploaded_image='/static/uploaded_image.png')

# Start the app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
