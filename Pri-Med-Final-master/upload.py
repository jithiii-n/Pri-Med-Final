import os
import shutil
import time
import numpy as np
import gc
from flask import Flask, Blueprint, request, redirect, render_template
from werkzeug.utils import secure_filename
import tenseal as ts
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Flask app setup
UPLOAD_FOLDER = os.path.abspath('static/uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
upload_bp = Blueprint('upload', __name__)

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class HECNN(nn.Module):
    def __init__(self):
        super(HECNN, self).__init__()
        self.fc1 = nn.Linear(784, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = x * x  # Square activation for HE compatibility
        x = self.fc2(x)
        return x


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def safe_clear_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        max_attempts = 3
        attempt = 0
        while attempt < max_attempts:
            try:
                if os.path.isfile(file_path):
                    # Close any open file handles
                    gc.collect()
                    os.close(os.open(file_path, os.O_RDONLY))
                    os.remove(file_path)
                break
            except Exception as e:
                print(f"Attempt {attempt + 1}: Error deleting file {file_path}: {e}")
                attempt += 1
                time.sleep(0.1)  # Wait briefly before retrying


class HEImagePipeline:
    def __init__(self):
        self.model = HECNN()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.load_model()
        self.context = self.generate_keys()

    def load_model(self):
        model_path = "models/new_chest_model.pth"
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print("Model loaded successfully from models/ directory.")
        else:
            print("Model file not found in models/ directory!")

    def generate_keys(self):
        params = {
            "scheme": ts.SCHEME_TYPE.CKKS,
            "poly_modulus_degree": 16384,
            "coeff_mod_bit_sizes": [60, 40, 40, 40, 40, 40, 40, 60]
        }
        context = ts.context(**params)
        context.global_scale = 2 ** 40
        context.generate_galois_keys()
        return context

    def preprocess_image(self, image):
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        image = transform(image)
        image = image.view(-1).cpu().numpy()  # Flatten and convert to numpy
        return image

    def encrypt_data(self, data):
        """Encrypt the input data using CKKS scheme"""
        try:
            # Normalize input data to a small range (e.g., [-1, 1])
            data_normalized = data / np.max(np.abs(data))
            encrypted_data = ts.ckks_vector(self.context, data_normalized.tolist())
            return encrypted_data
        except Exception as e:
            print(f"Error encrypting data: {str(e)}")
            raise

    def encrypted_inference(self, encrypted_data):
        """Perform inference on encrypted data"""
        try:
            # Move model weights and biases to CPU and convert to NumPy
            fc1_weight = self.model.fc1.weight.data.cpu().numpy()
            fc1_bias = self.model.fc1.bias.data.cpu().numpy()
            fc2_weight = self.model.fc2.weight.data.cpu().numpy()
            fc2_bias = self.model.fc2.bias.data.cpu().numpy()

            # Perform encrypted operations
            # Layer 1: fc1
            encrypted_output = encrypted_data.mm(fc1_weight.T) + fc1_bias
            encrypted_output = encrypted_output.polyval([0, 0, 1])  # Square activation

            # Layer 2: fc2
            encrypted_output = encrypted_output.mm(fc2_weight.T) + fc2_bias

            # Approximate sigmoid activation using a polynomial
            # sigmoid(x) ≈ 0.5 + 0.15 * x - 0.0015 * x^3
            encrypted_output = encrypted_output.polyval([0.5, 0.15, 0, -0.0015])

            return encrypted_output
        except Exception as e:
            print(f"Error during encrypted inference: {str(e)}")
            raise

    def decrypt_result(self, encrypted_result):
        try:
            return encrypted_result.decrypt()
        except Exception as e:
            print(f"Error decrypting result: {str(e)}")
            raise


# Initialize pipeline
pipeline = HEImagePipeline()


@upload_bp.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)

        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)

            # Clear folder before saving new file
            safe_clear_folder(UPLOAD_FOLDER)

            file.save(file_path)

            try:
                # Try to verify the image file integrity
                with Image.open(file_path) as img:
                    img.verify()  # Verify the file is intact

                # Reopen for processing since verify() closes the file
                image = Image.open(file_path).convert('L')

            except Exception as img_error:
                return render_template('result.html',
                                       output=f"Invalid or corrupted image file: {str(img_error)}")

            preprocessed_image = pipeline.preprocess_image(image)
            encrypted_image = pipeline.encrypt_data(preprocessed_image)
            encrypted_result = pipeline.encrypted_inference(encrypted_image)
            decrypted_result = pipeline.decrypt_result(encrypted_result)

            # Get prediction
            prediction = "Pneumonia detected" if decrypted_result[0] > 0.5 else "Congrats, Youre Healthy"

            # Close image explicitly
            image.close()

            return render_template('result.html',
                                   output=f"Predicted class: {prediction}",
                                   confidence=f"Confidence: {abs(decrypted_result[0] - 0.5) * 2:.2f}")

        except Exception as e:
            print(f"Error processing image: {e}")
            return render_template('result.html', output=f"Error during file processing: {str(e)}")
        finally:
            # Try to clean up in the finally block
            try:
                if 'image' in locals():
                    image.close()
            except:
                pass

    return render_template('index.html')


if __name__ == '__main__':
    app.register_blueprint(upload_bp)
    app.run(debug=True)