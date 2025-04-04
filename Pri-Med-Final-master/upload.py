import os
import time
import numpy as np
import gc
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import tenseal as ts
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

# Create a Blueprint for the upload functionality
upload_bp = Blueprint('upload', __name__)

# Flask app setup
UPLOAD_FOLDER = os.path.abspath('static/uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global state to track progress
upload_bp.config = {
    'UPLOAD_FOLDER': UPLOAD_FOLDER,
    'IMAGE_UPLOADED': False,
    'KEYS_GENERATED': False,
    'IMAGE_ENCRYPTED': False,
    'INFERENCE_DONE': False
}


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
    """Clear all files in the specified folder."""
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
        self.context = None  # Initialize context as None

    def load_model(self):
        model_path = "models/new_chest_model.pth"  # Keeping original model path
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print("Model loaded successfully from models/ directory.")
        else:
            print("Model file not found in models/ directory!")

    def generate_keys(self):
        try:
            params = {
                "scheme": ts.SCHEME_TYPE.CKKS,
                "poly_modulus_degree": 16384,
                "coeff_mod_bit_sizes": [60, 40, 40, 40, 40, 40, 40, 60]
            }
            context = ts.context(**params)
            context.global_scale = 2 ** 40
            context.generate_galois_keys()
            self.context = context  # Store the context
            return context
        except Exception as e:
            print(f"Error generating keys: {str(e)}")
            return None

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

    def save_encrypted_data(self, encrypted_data, file_path):
        """Save encrypted data to a file"""
        try:
            with open(file_path, 'wb') as f:
                f.write(encrypted_data.serialize())
        except Exception as e:
            print(f"Error saving encrypted data: {str(e)}")
            raise

    def load_encrypted_data(self, file_path):
        """Load encrypted data from a file"""
        try:
            with open(file_path, 'rb') as f:
                serialized_data = f.read()
            return ts.ckks_vector_from(self.context, serialized_data)
        except Exception as e:
            print(f"Error loading encrypted data: {str(e)}")
            raise

    def encrypted_inference(self, encrypted_data):
        """Perform inference on encrypted data"""
        try:
            start_time = time.time()  # Start timing

            # Move model weights and biases to CPU and convert to NumPy
            fc1_weight = self.model.fc1.weight.data.cpu().numpy()
            fc1_bias = self.model.fc1.bias.data.cpu().numpy()
            fc2_weight = self.model.fc2.weight.data.cpu().numpy()
            fc2_bias = self.model.fc2.bias.data.cpu().numpy()

            # Perform encrypted operations
            print("Starting encrypted inference...")

            # Layer 1: fc1
            encrypted_output = encrypted_data.mm(fc1_weight.T) + fc1_bias
            encrypted_output = encrypted_output.polyval([0, 0, 1])  # Square activation
            print("Layer 1 completed.")

            # Layer 2: fc2
            encrypted_output = encrypted_output.mm(fc2_weight.T) + fc2_bias
            print("Layer 2 completed.")

            # Approximate sigmoid activation using a polynomial
            # sigmoid(x) ≈ 0.5 + 0.15 * x - 0.0015 * x^3
            encrypted_output = encrypted_output.polyval([0.5, 0.15, 0, -0.0015])
            print("Sigmoid approximation applied.")

            end_time = time.time()  # End timing
            inference_time = end_time - start_time
            print(f"Encrypted inference completed in {inference_time:.4f} seconds.")

            return encrypted_output, inference_time
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


@upload_bp.route('/upload', methods=['POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename == '' or not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'No file selected or file type not allowed'}), 400

        try:
            # Clear the uploads directory
            safe_clear_folder(upload_bp.config['UPLOAD_FOLDER'])

            # Generate keys
            pipeline.generate_keys()
            if pipeline.context is None:
                return jsonify({'success': False, 'error': 'Failed to generate encryption keys'}), 500

            # Preprocess and encrypt the image
            image = Image.open(file.stream).convert('L')  # Convert to grayscale
            preprocessed_image = pipeline.preprocess_image(image)
            encrypted_image = pipeline.encrypt_data(preprocessed_image)

            # Save the encrypted image
            encrypted_file_path = os.path.join(upload_bp.config['UPLOAD_FOLDER'], "encrypted_image.enc")
            pipeline.save_encrypted_data(encrypted_image, encrypted_file_path)

            # Update state
            upload_bp.config['IMAGE_UPLOADED'] = True
            upload_bp.config['KEYS_GENERATED'] = True
            upload_bp.config['IMAGE_ENCRYPTED'] = True

            return jsonify({'success': True, 'message': 'Image uploaded and encrypted successfully'})

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    return jsonify({'success': False, 'error': 'Invalid request method'}), 400


@upload_bp.route('/inference', methods=['POST'])
def inference():
    try:
        # Load the encrypted image
        encrypted_file_path = os.path.join(upload_bp.config['UPLOAD_FOLDER'], "encrypted_image.enc")
        encrypted_image = pipeline.load_encrypted_data(encrypted_file_path)

        # Perform encrypted inference
        encrypted_result, inference_time = pipeline.encrypted_inference(encrypted_image)

        # Save the encrypted result
        result_file_path = os.path.join(upload_bp.config['UPLOAD_FOLDER'], "encrypted_result.enc")
        pipeline.save_encrypted_data(encrypted_result, result_file_path)

        # Update state
        upload_bp.config['INFERENCE_DONE'] = True

        # Return inference time in the response
        return jsonify({
            'success': True,
            'message': 'Inference completed successfully',
            'inference_time': f"{inference_time:.4f}"  # Format to 4 decimal places
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@upload_bp.route('/decrypt', methods=['POST'])
def decrypt():
    try:
        # Load the encrypted result
        result_file_path = os.path.join(upload_bp.config['UPLOAD_FOLDER'], "encrypted_result.enc")
        encrypted_result = pipeline.load_encrypted_data(result_file_path)

        # Decrypt the result
        decrypted_result = pipeline.decrypt_result(encrypted_result)

        # Get prediction
        prediction = "Pneumonia detected" if decrypted_result[0] > 0.5 else "Congrats, Youre Healthy"
        confidence = f"Confidence: {abs(decrypted_result[0] - 0.5) * 2:.2f}"

        return jsonify({
            'success': True,
            'prediction': prediction,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@upload_bp.route('/result')
def result():
    prediction = request.args.get('prediction', 'No prediction available')
    confidence = request.args.get('confidence', 'No confidence score available')
    return render_template('result.html', output=f"{prediction} ({confidence})")