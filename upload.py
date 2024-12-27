import torch
import tenseal as ts
from torchvision import transforms
from PIL import Image
from flask import Flask, Blueprint, request, redirect, render_template
from werkzeug.utils import secure_filename
import os
import pickle
import shutil
import torch.nn as nn

# Define label mappings for BreastMNIST dataset
LABELS = {0: 'Benign', 1: 'Malignant'}

# Model definition
class BreastMNISTModel(nn.Module):
    def __init__(self):
        super(BreastMNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.avg_pool(x)
        x = self.relu(self.conv2(x))
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Encryption utilities
def generate_keys():
    try:
        bits_scale = 26
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
        )
        context.global_scale = pow(2, bits_scale)
        context.generate_galois_keys()

        loc = os.path.abspath('static/uploads/')
        os.makedirs(loc, exist_ok=True)
        prvt_key_loc = os.path.join(loc, 'private_key_ctx.pickle')
        pbl_key_loc = os.path.join(loc, 'public_key_ctx.pickle')

        with open(prvt_key_loc, 'wb') as handle:
            pickle.dump(context.serialize(save_secret_key=True), handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(pbl_key_loc, 'wb') as handle:
            pickle.dump(context.serialize(save_secret_key=False), handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("Keys successfully generated.")
        return prvt_key_loc, pbl_key_loc
    except Exception as e:
        print(f"Error generating keys: {e}")
        return None, None

def load_keys():
    loc = os.path.abspath('static/uploads/')
    prvt_key_loc = os.path.join(loc, 'private_key_ctx.pickle')
    pbl_key_loc = os.path.join(loc, 'public_key_ctx.pickle')

    if not os.path.exists(prvt_key_loc) or not os.path.exists(pbl_key_loc):
        print("Keys not found. Regenerating...")
        return generate_keys()
    print("Keys loaded successfully.")
    return prvt_key_loc, pbl_key_loc

def encrypt_tensor(img_tensor, public_key_serialized):
    try:
        print(f"Original tensor shape: {img_tensor.shape}")
        context = ts.context_from(public_key_serialized)
        flat_tensor = img_tensor.view(-1).numpy().tolist()
        encrypted_vector = ts.ckks_vector(context, flat_tensor)
        print(f"Encrypted vector length: {len(flat_tensor)}")
        return encrypted_vector
    except Exception as e:
        print(f"Error during encryption: {e}")
        return None


def decrypt_tensor(enc_tensor_serialized, private_key_serialized):
    try:
        context = ts.context_from(private_key_serialized)
        ckks_vector = ts.ckks_vector_from(context, enc_tensor_serialized)
        decrypted = ckks_vector.decrypt()
        decrypted_tensor = torch.tensor(decrypted).view(1, 1, 28, 28)
        print(f"Decrypted tensor shape: {decrypted_tensor.shape}")
        return decrypted_tensor
    except Exception as e:
        print(f"Error during decryption: {e}")
        return None


# Load model
def load_model():
    try:
        model_path = os.path.abspath('models/breastmnist_model.pth')
        model = BreastMNISTModel()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Flask app setup
UPLOAD_FOLDER = os.path.abspath('static/uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def safe_clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

upload_bp = Blueprint('upload', __name__)

@upload_bp.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        safe_clear_folder(UPLOAD_FOLDER)
        file = request.files.get('image')
        if not file or file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)

        try:
            img = Image.open(file.stream)
            transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            img_tensor = transform(img).unsqueeze(0)

            private_key_path, public_key_path = load_keys()
            with open(public_key_path, 'rb') as handle:
                public_key_serialized = pickle.load(handle)
            with open(private_key_path, 'rb') as handle:
                private_key_serialized = pickle.load(handle)

            encrypted_img_tensor = encrypt_tensor(img_tensor, public_key_serialized)
            enc_tensor_path = os.path.join(UPLOAD_FOLDER, 'encrypted_img_tensor.pickle')
            with open(enc_tensor_path, 'wb') as handle:
                pickle.dump(encrypted_img_tensor.serialize(), handle)

            with open(enc_tensor_path, 'rb') as handle:
                encrypted_img_data = pickle.load(handle)

            decrypted_img_tensor = decrypt_tensor(encrypted_img_data, private_key_serialized)
            torch_model = load_model()
            output = torch_model(decrypted_img_tensor)
            predicted_label_idx = torch.argmax(output, dim=1).item()
            predicted_label = LABELS.get(predicted_label_idx, "Unknown")

            return render_template('result.html', output=predicted_label)
        except Exception as e:
            print(f"Error processing image: {e}")


    return render_template('index.html')
