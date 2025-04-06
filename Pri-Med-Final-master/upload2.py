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

# Create a Blueprint for the upload2 functionality
upload2_bp = Blueprint('upload2', __name__)

# Flask app setup
UPLOAD2_FOLDER = os.path.abspath('static/upload2s')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure the upload2 folder exists
os.makedirs(UPLOAD2_FOLDER, exist_ok=True)

# Global state to track progress
upload2_bp.config = {
    'UPLOAD2_FOLDER': UPLOAD2_FOLDER,
    'IMAGE_UPLOAD2ED': False,
    'KEYS_GENERATED': False,
    'IMAGE_ENCRYPTED': False,
    'INFERENCE_DONE': False
}


# Alzheimer's Model Definition
class HECNN(nn.Module):
    def __init__(self):
        super(HECNN, self).__init__()
        self.fc1 = nn.Linear(784, 32)
        self.fc2 = nn.Linear(32, 4)  # 4 output classes for Alzheimer's
        self.activation = lambda x: x.pow(2)  # Square activation

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
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
                    gc.collect()
                    os.close(os.open(file_path, os.O_RDONLY))
                    os.remove(file_path)
                break
            except Exception as e:
                print(f"Attempt {attempt + 1}: Error deleting file {file_path}: {e}")
                attempt += 1
                time.sleep(0.1)


class HEImagePipeline:
    def __init__(self):
        self.model = HECNN()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.load_model()
        self.context = None
        self.class_names = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']

    def load_model(self):
        model_path = "models/mri (3).pth"  # Updated model path
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print("Alzheimer's model loaded successfully")
        else:
            print("Alzheimer's model file not found!")

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
            self.context = context
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
        image = image.view(-1).cpu().numpy()
        return image

    def encrypt_data(self, data):
        try:
            data_normalized = data / np.max(np.abs(data))
            encrypted_data = ts.ckks_vector(self.context, data_normalized.tolist())
            return encrypted_data
        except Exception as e:
            print(f"Error encrypting data: {str(e)}")
            raise

    def save_encrypted_data(self, encrypted_data, file_path):
        try:
            with open(file_path, 'wb') as f:
                f.write(encrypted_data.serialize())
        except Exception as e:
            print(f"Error saving encrypted data: {str(e)}")
            raise

    def load_encrypted_data(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                serialized_data = f.read()
            return ts.ckks_vector_from(self.context, serialized_data)
        except Exception as e:
            print(f"Error loading encrypted data: {str(e)}")
            raise

    def encrypted_inference(self, encrypted_data):
        try:
            start_time = time.time()

            # Get model parameters
            fc1_weight = self.model.fc1.weight.data.cpu().numpy()
            fc1_bias = self.model.fc1.bias.data.cpu().numpy()
            fc2_weight = self.model.fc2.weight.data.cpu().numpy()
            fc2_bias = self.model.fc2.bias.data.cpu().numpy()

            # Layer 1
            encrypted_output = encrypted_data.mm(fc1_weight.T) + fc1_bias
            encrypted_output = encrypted_output.square()  # x² activation

            # Layer 2
            encrypted_output = encrypted_output.mm(fc2_weight.T) + fc2_bias

            end_time = time.time()
            inference_time = end_time - start_time
            print(f"Encrypted inference completed in {inference_time:.4f} seconds")

            return encrypted_output, inference_time
        except Exception as e:
            print(f"Error during encrypted inference: {str(e)}")
            raise

    def decrypt_result(self, encrypted_result):
        try:
            decrypted = encrypted_result.decrypt()
            probabilities = torch.softmax(torch.tensor(decrypted), dim=0)
            return probabilities.numpy()
        except Exception as e:
            print(f"Error decrypting result: {str(e)}")
            raise


# Initialize pipeline
pipeline = HEImagePipeline()


@upload2_bp.route('/upload2', methods=['POST'])
def upload2_image():
    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename == '' or not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'No file selected or file type not allowed'}), 400

        try:
            safe_clear_folder(upload2_bp.config['UPLOAD2_FOLDER'])

            # Generate keys
            pipeline.generate_keys()
            if pipeline.context is None:
                return jsonify({'success': False, 'error': 'Failed to generate encryption keys'}), 500

            # Process image
            image = Image.open(file.stream).convert('L')
            preprocessed_image = pipeline.preprocess_image(image)
            encrypted_image = pipeline.encrypt_data(preprocessed_image)

            # Save encrypted image
            encrypted_file_path = os.path.join(upload2_bp.config['UPLOAD2_FOLDER'], "encrypted_image.enc")
            pipeline.save_encrypted_data(encrypted_image, encrypted_file_path)

            upload2_bp.config.update({
                'IMAGE_UPLOAD2ED': True,
                'KEYS_GENERATED': True,
                'IMAGE_ENCRYPTED': True
            })

            return jsonify({'success': True, 'message': 'Image upload2ed and encrypted successfully'})

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    return jsonify({'success': False, 'error': 'Invalid request method'}), 400


@upload2_bp.route('/inference', methods=['POST'])
def inference():
    try:
        encrypted_file_path = os.path.join(upload2_bp.config['UPLOAD2_FOLDER'], "encrypted_image.enc")
        encrypted_image = pipeline.load_encrypted_data(encrypted_file_path)

        encrypted_result, inference_time = pipeline.encrypted_inference(encrypted_image)

        result_file_path = os.path.join(upload2_bp.config['UPLOAD2_FOLDER'], "encrypted_result.enc")
        pipeline.save_encrypted_data(encrypted_result, result_file_path)

        upload2_bp.config['INFERENCE_DONE'] = True

        return jsonify({
            'success': True,
            'message': 'Inference completed successfully',
            'inference_time': f"{inference_time:.4f}"
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@upload2_bp.route('/decrypt', methods=['POST'])
def decrypt():
    try:
        result_file_path = os.path.join(upload2_bp.config['UPLOAD2_FOLDER'], "encrypted_result.enc")
        encrypted_result = pipeline.load_encrypted_data(result_file_path)

        probabilities = pipeline.decrypt_result(encrypted_result)
        predicted_class = np.argmax(probabilities)

        return jsonify({
            'success': True,
            'prediction': pipeline.class_names[predicted_class],
            'confidence': f"{probabilities[predicted_class] * 100:.2f}%",
            'probabilities': {name: float(prob) for name, prob in zip(pipeline.class_names, probabilities)}
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@upload2_bp.route('/result')
def result():
    prediction = request.args.get('prediction', 'No prediction available')
    confidence = request.args.get('confidence', 'No confidence score available')
    return render_template('result.html',
                           output=f"{prediction} ({confidence})",
                           class_names=pipeline.class_names,
                           probabilities=request.args.get('probabilities', {}))