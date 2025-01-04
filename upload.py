import tenseal as ts
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import time
import os
import pickle
import shutil
from flask import Flask, Blueprint, request, redirect, render_template
from werkzeug.utils import secure_filename


# Define MNIST Labels
MNIST_LABELS = {0: 'Zero', 1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine'}

# Define ConvNet class (your model class)
class ConvNet(torch.nn.Module):
    def __init__(self, hidden=64, output=10):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=7, padding=0, stride=3)
        self.fc1 = torch.nn.Linear(256, hidden)
        self.fc2 = torch.nn.Linear(hidden, output)

    def forward(self, x):
        x = self.conv1(x)
        # the model uses the square activation function
        x = x * x
        # flattening while keeping the batch axis
        x = x.view(-1, 256)
        x = self.fc1(x)
        x = x * x
        x = self.fc2(x)
        return x

# Define EncConvNet class for encrypted model inference
class EncConvNet:
    def __init__(self, torch_nn):
        self.conv1_weight = torch_nn.conv1.weight.data.view(
            torch_nn.conv1.out_channels, torch_nn.conv1.kernel_size[0],
            torch_nn.conv1.kernel_size[1]
        ).tolist()
        self.conv1_bias = torch_nn.conv1.bias.data.tolist()

        self.fc1_weight = torch_nn.fc1.weight.T.data.tolist()
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()

        self.fc2_weight = torch_nn.fc2.weight.T.data.tolist()
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()

    def forward(self, enc_x, windows_nb):
        # Convolutional layer
        start = time.time()
        enc_channels = []
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            y = enc_x.conv2d_im2col(kernel, windows_nb) + bias
            enc_channels.append(y)
        end = time.time()
        print("Conv2D takes", end - start)

        # Pack all channels into a single flattened vector
        enc_x = ts.CKKSVector.pack_vectors(enc_channels)

        # Square activation
        enc_x.square_()

        # Fully connected layer 1
        start = time.time()
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        end = time.time()
        print("FC1 takes", end - start)

        # Square activation
        enc_x.square_()

        # Fully connected layer 2
        start = time.time()
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        end = time.time()
        print("FC2 takes", end - start)

        return enc_x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# Encryption utilities
def create_context(bits_scale=26):
    """Create and return a TenSEAL encryption context."""
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
    )
    context.global_scale = pow(2, bits_scale)
    context.generate_galois_keys()  # Required for ciphertext rotations
    return context


def preprocess_image(image_path):
    """Preprocess the input image to match the MNIST format."""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image_tensor = transform(image).view(-1)  # Flatten the image
    return image_tensor


def encrypt_image(image_tensor, context):
    """Encrypt the preprocessed image using TenSEAL."""
    return ts.ckks_vector(context, image_tensor.numpy())


# Flask app setup
UPLOAD_FOLDER = os.path.abspath('static/uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def safe_clear_folder(folder_path):
    """Clear the upload folder."""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")


# Main pipeline for image processing, encryption, and model inference
def main_pipeline(image_path, enc_model, kernel_shape, stride, bits_scale=26):
    """Run the pipeline: preprocess, encode, encrypt, and evaluate."""
    print("Creating TenSEAL context...")
    context = create_context(bits_scale)

    print("Loading the ConvNet model...")
    model = torch.load("models/model.pth")
    model.eval()

    print("Converting to encrypted model (EncConvNet)...")
    enc_model = EncConvNet(model)

    print("Preprocessing the image...")
    image_tensor = preprocess_image(image_path)
    print("Shape of image tensor before encoding:", image_tensor.shape)

    print("Encoding and encrypting the image...")
    x_enc, windows_nb = ts.im2col_encoding(
        context, image_tensor.view(28, 28).tolist(), kernel_shape[0], kernel_shape[1], stride
    )
    print("Number of encoded windows:", windows_nb)

    print("Running encrypted evaluation...")
    try:
        start_time = time.time()
        enc_output = enc_model(x_enc, windows_nb)
        elapsed_time = time.time() - start_time
        print(f"Encrypted evaluation completed in {elapsed_time:.2f} seconds.")

        print("Decrypting the result...")
        output = enc_output.decrypt()
        output = torch.tensor(output).view(1, -1)

        print("Result:", output)
        _, predicted = torch.max(output, 1)
        predicted_label = MNIST_LABELS.get(predicted.item(), "Unknown")
        print(f"Predicted class: {predicted_label}")
        return predicted_label
    except Exception as e:
        print("Error during model inference:", e)


# Flask app for file upload and result rendering
app = Flask(__name__)
upload_bp = Blueprint('upload', __name__)

@upload_bp.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        safe_clear_folder(UPLOAD_FOLDER)
        file = request.files.get('image')
        if not file or file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)

        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            # Call the main pipeline
            model = torch.load("models/model.pth")

            # Get kernel shape and stride from the loaded model
            kernel_shape = model.conv1.kernel_size
            stride = model.conv1.stride[0]

            # Initialize the encrypted model
            enc_model = EncConvNet(model)

            # Run the main pipeline
            predicted_label = main_pipeline(file_path, enc_model, kernel_shape, stride)

            if predicted_label:
                return render_template('result.html', output=f"Predicted class: {predicted_label}")
            else:
                return render_template('result.html', output="Error during prediction.")
        except Exception as e:
            print(f"Error processing image: {e}")

    return render_template('index.html')
