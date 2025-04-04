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
upload2_bp = Blueprint('upload2', __name__)

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class HECNN(nn.Module):
    """
    Homomorphic Encryption-compatible CNN model with Chebyshev approximation of ReLU activation.
    """

    def __init__(self):
        super(HECNN, self).__init__()
        self.fc1 = nn.Linear(8112, 64)  # Input size adjusted for 52x52 RGB images
        self.fc2 = nn.Linear(64, 1)

    def chebyshev_activation(self, x):
        """
        Chebyshev approximation of ReLU: f(x) = c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4 + c5*x^5
        Coefficients: [0.5, 0.25, -0.0208333, 0.00260417, -0.000434028, 0.0000826823]
        """
        coeffs = [0.5, 0.25, -0.0208333, 0.00260417, -0.000434028, 0.0000826823]
        result = coeffs[0]
        x_power = x.clone()  # Avoid inplace modification
        for coeff in coeffs[1:]:
            result += coeff * x_power
            x_power = x_power * x  # Avoid inplace operation
        return result

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc1(x)
        x = self.chebyshev_activation(x)  # Use Chebyshev activation
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
    """
    Pipeline for training, encrypting, and performing inference on images.
    """

    def __init__(self):
        self.model = HECNN()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.load_model()
        self.context = self.generate_keys()

    def load_model(self):
        """Load the pre-trained model."""
        model_path = "models/rgbmodelnew2.pth"
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print("Model loaded successfully from models/ directory.")
        else:
            print("Model file not found in models/ directory!")

    def generate_keys(self):
        """Generate encryption keys for the CKKS scheme."""
        params = {
            "scheme": ts.SCHEME_TYPE.CKKS,
            "poly_modulus_degree": 16384,
            "coeff_mod_bit_sizes": [60, 40, 40, 40, 40, 40, 40, 60],
        }
        context = ts.context(**params)
        context.global_scale = 2**40
        context.generate_galois_keys()
        return context

    def preprocess_image(self, image):
        """Preprocess the image for inference."""
        transform = transforms.Compose([
            transforms.Resize((52, 52)),  # Resize to 52x52
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])
        image = transform(image).unsqueeze(0)  # Add batch dimension
        return image

    def encrypt_data(self, data):
        """Encrypt input data."""
        try:
            data_normalized = data / np.max(np.abs(data))
            data_flattened = data_normalized.flatten()
            encrypted_data = ts.ckks_vector(self.context, data_flattened.tolist())
            return encrypted_data
        except Exception as e:
            print(f"Error encrypting data: {str(e)}")
            raise

    def save_encrypted_data(self, encrypted_data, file_path):
        """Save encrypted data to a file."""
        try:
            with open(file_path, "wb") as f:
                f.write(encrypted_data.serialize())
        except Exception as e:
            print(f"Error saving encrypted data: {str(e)}")
            raise

    def load_encrypted_data(self, file_path):
        """Load encrypted data from a file."""
        try:
            with open(file_path, "rb") as f:
                serialized_data = f.read()
            return ts.ckks_vector_from(self.context, serialized_data)
        except Exception as e:
            print(f"Error loading encrypted data: {str(e)}")
            raise

    def encrypted_inference(self, encrypted_data):
        """Perform inference on encrypted data."""
        try:
            # Extract model weights and biases
            fc1_weight = self.model.fc1.weight.data.cpu().numpy()
            fc1_bias = self.model.fc1.bias.data.cpu().numpy()
            fc2_weight = self.model.fc2.weight.data.cpu().numpy()
            fc2_bias = self.model.fc2.bias.data.cpu().numpy()

            # Layer 1: fc1
            encrypted_output = encrypted_data.mm(fc1_weight.T) + fc1_bias

            # Apply Chebyshev approximation of ReLU
            encrypted_output = encrypted_output.polyval([0.5, 0.25, -0.0208333, 0.00260417, -0.000434028, 0.0000826823])

            # Layer 2: fc2
            encrypted_output = encrypted_output.mm(fc2_weight.T) + fc2_bias

            return encrypted_output
        except Exception as e:
            print(f"Error during encrypted inference: {str(e)}")
            raise

    def decrypt_result(self, encrypted_result):
        """Decrypt the result of encrypted inference."""
        try:
            decrypted_result = encrypted_result.decrypt()
            return decrypted_result
        except Exception as e:
            print(f"Error decrypting result: {str(e)}")
            raise


# Initialize pipeline
pipeline = HEImagePipeline()


@upload2_bp.route('/upload2', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)

        try:
            # Clear folder before saving new file
            safe_clear_folder(UPLOAD_FOLDER)

            # Process the image in memory
            with Image.open(file.stream) as img:
                img.verify()  # Verify the file is intact
                img = Image.open(file.stream).convert('RGB')  # Convert to RGB

                # Preprocess the image
                preprocessed_image = pipeline.preprocess_image(img)

                # Encrypt the preprocessed image
                encrypted_image = pipeline.encrypt_data(preprocessed_image.numpy())

                # Save the encrypted image
                filename = secure_filename(file.filename)
                encrypted_file_path = os.path.join(UPLOAD_FOLDER, f"{filename}.enc")
                pipeline.save_encrypted_data(encrypted_image, encrypted_file_path)

                # Perform inference on the encrypted image
                loaded_encrypted_image = pipeline.load_encrypted_data(encrypted_file_path)
                encrypted_result = pipeline.encrypted_inference(loaded_encrypted_image)
                decrypted_result = pipeline.decrypt_result(encrypted_result)

                # Get prediction and confidence level
                prediction = "Congrats, You're Healthy" if decrypted_result[0] > 0.05 else "CANCER"
                confidence_level = abs(decrypted_result[0])  # Confidence is the absolute value of the result

                # Debugging: Print confidence level to terminal
                print(f"Debug: Confidence Level = {confidence_level:.2f}")

                return render_template('result.html',
                                      output=f"Predicted class: {prediction}",
                                      confidence=f"Confidence: {confidence_level:.2f}")

        except Exception as e:
            print(f"Error processing image: {e}")
            return render_template('result.html', output=f"Error during file processing: {str(e)}")

    return render_template('index.html')


if __name__ == '__main__':
    app.register_blueprint(upload2_bp)
    app.run(debug=True)