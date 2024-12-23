import tenseal as ts
import torch
from torchvision import transforms
from PIL import Image
from flask import Blueprint, request, redirect, render_template
import os
from werkzeug.utils import secure_filename
import io

# Initialize the Blueprint for file upload
upload_bp = Blueprint('upload', __name__)

# Set up the file upload configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Image transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),  # Resize image to 28x28
    transforms.ToTensor(),
])

# Encryption Parameters
bits_scale = 26  # Controls precision of the fractional part

# Encryption setup function
def initialize_tenseal_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
    )
    context.global_scale = pow(2, bits_scale)
    context.generate_galois_keys()
    return context

# Allowed file type check
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Model loading function
def load_model():
    model = torch.load('models/model.h5')
    model.eval()  # Set model to evaluation mode
    return model

# Encrypt and save data
def encrypt_and_save_data(img_tensor, context, filename):
    encrypted_img_tensor, _ = ts.im2col_encoding(
        context, img_tensor.view(28, 28).tolist(), 3, 3, 1  # Assuming 3x3 kernel
    )

    encrypted_img_path = os.path.join(UPLOAD_FOLDER, filename + ".enc")
    with open(encrypted_img_path, "wb") as enc_file:
        enc_file.write(encrypted_img_tensor.serialize())

    context_file_path = os.path.join(UPLOAD_FOLDER, filename + ".context")
    with open(context_file_path, "wb") as context_file:
        context_file.write(context.serialize())

# Function for homomorphic evaluation (prediction)
def predict_encrypted(model, encrypted_img_tensor, context):
    # Perform encrypted evaluation
    # Assuming the model is adapted for encrypted tensors
    encrypted_prediction = model(encrypted_img_tensor)  # Example placeholder
    decrypted_prediction = context.decrypt(encrypted_prediction)
    predicted_label = torch.argmax(decrypted_prediction).item()
    return predicted_label

# Route for handling image upload and prediction
@upload_bp.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # Secure the filename
            filename = secure_filename(file.filename)

            # Open the image without saving it first
            img = Image.open(file.stream)  # Read directly from the stream
            img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

            # Initialize the context for encrypted operations
            context = initialize_tenseal_context()

            # Encrypt and save the encrypted image data
            encrypt_and_save_data(img_tensor, context, filename)

            # Load the pre-trained model
            model = load_model()

            # Predict on encrypted data
            # Predict on encrypted data
            predicted_label = predict_encrypted(model, img_tensor, context)

            # Map prediction to proper labels for binary classification
            label_mapping = {0: 'benign', 1: 'malignant'}
            predicted_class = label_mapping.get(predicted_label, "unknown")

            return render_template('about.html', predicted_label=predicted_class)
