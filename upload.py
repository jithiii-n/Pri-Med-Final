import torch
import torch.nn as nn
import tenseal as ts
from torchvision import transforms
from PIL import Image
from flask import Flask, Blueprint, request, redirect, render_template
from werkzeug.utils import secure_filename

# Define the 8-layer convolutional neural network model
class ConvNet(nn.Module):
    def __init__(self, hidden=64, output=2):  # Binary output (benign vs. malignant)
        super(ConvNet, self).__init__()

        # Convolutional layers (3 layers)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # 3 input channels (RGB)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # MaxPooling layers (2 layers)
        self.pool1 = nn.MaxPool2d(2, 2)  # Reduces size by half
        self.pool2 = nn.MaxPool2d(2, 2)

        # Flatten layer before fully connected layer
        self.flatten = nn.Flatten()

        # Compute output size from convolution and pooling layers
        self.fc1_input_size = 128 * 32 * 32  # After two max pooling layers (128x128 -> 32x32)

        # Fully connected layers (2 layers)
        self.fc1 = nn.Linear(self.fc1_input_size, hidden)  # Adjusted to match output size after conv and pooling
        self.fc2 = nn.Linear(hidden, output)  # Output 2 classes: benign, malignant

    def piecewise_linear(self, x):
        """Piecewise linear activation function."""
        return torch.where(x < 0, 0.1 * x, x)  # Example: Linear in positive region, scaled negative region

    def forward(self, x):
        # Forward pass through convolutional and pooling layers
        x = self.pool1(self.piecewise_linear(self.conv1(x)))
        x = self.pool2(self.piecewise_linear(self.conv2(x)))
        x = self.pool2(self.piecewise_linear(self.conv3(x)))  # Applying second pooling after third conv

        # Flatten the output for fully connected layers
        x = self.flatten(x)

        # Fully connected layers with piecewise linear activation
        x = self.piecewise_linear(self.fc1(x))
        x = self.fc2(x)
        return x

# Encapsulate the model with encryption logic
class EncConvNet:
    def __init__(self, torch_nn):
        # Extract weights and biases for homomorphic encryption
        self.conv1_weight = torch_nn.conv1.weight.data.view(
            torch_nn.conv1.out_channels, torch_nn.conv1.kernel_size[0], torch_nn.conv1.kernel_size[1]
        ).tolist()
        self.conv1_bias = torch_nn.conv1.bias.data.tolist()

        self.conv2_weight = torch_nn.conv2.weight.data.view(
            torch_nn.conv2.out_channels, torch_nn.conv2.kernel_size[0], torch_nn.conv2.kernel_size[1]
        ).tolist()
        self.conv2_bias = torch_nn.conv2.bias.data.tolist()

        self.conv3_weight = torch_nn.conv3.weight.data.view(
            torch_nn.conv3.out_channels, torch_nn.conv3.kernel_size[0], torch_nn.conv3.kernel_size[1]
        ).tolist()
        self.conv3_bias = torch_nn.conv3.bias.data.tolist()

        self.fc1_weight = torch_nn.fc1.weight.T.data.tolist()
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()

        self.fc2_weight = torch_nn.fc2.weight.T.data.tolist()
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()

    def forward(self, enc_x, context, conv_layers):
        enc_channels = []
        input_tensor = enc_x

        # Perform convolutions on encrypted input (using homomorphic operations)
        for layer_idx, (kernel, bias) in enumerate(conv_layers):
            kernel_shape = kernel.shape
            stride = 1  # Assuming stride is 1 for simplicity

            # Perform im2col encoding
            input_tensor = ts.im2col_encoding(
                context, input_tensor.tolist(), kernel_shape[0], kernel_shape[1], stride
            )

            # Homomorphic convolution (im2col encoding and then applying kernel)
            conv_output = input_tensor * kernel + bias
            enc_channels.append(conv_output)

        # Concatenate the channels from each convolution layer
        enc_x = ts.CKKSVector.pack_vectors(enc_channels)
        enc_x = enc_x.reshape(enc_x.size(0), -1)  # Flatten for fully connected layer

        # Fully connected layers (matrix multiplication)
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        enc_x.square_()  # Homomorphic square operation
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias

        return enc_x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# Encrypt image tensor
def encrypt_tensor(img_tensor, context):
    # Reshaping tensor and ensuring it's contiguous before encryption
    img_tensor = img_tensor.reshape(-1)  # Flatten tensor
    return ts.ckks_vector(context, img_tensor.tolist())

# Initialize encryption parameters
def initialize_tenseal_context():
    bits_scale = 26
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
    )
    context.global_scale = pow(2, bits_scale)
    context.generate_galois_keys()
    return context

# Load the model
def load_model():
    model = ConvNet()
    model.load_state_dict(torch.load('models/model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model


# Flask app setup (this is only for one Blueprint, app is initialized elsewhere)
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Allowable file check
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Create a Blueprint for the upload functionality
upload_bp = Blueprint('upload', __name__)

@upload_bp.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)

        filename = secure_filename(file.filename)
        img = Image.open(file.stream)
        transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
        img_tensor = transform(img).unsqueeze(0)

        context = initialize_tenseal_context()
        encrypted_img_tensor = encrypt_tensor(img_tensor, context)

        # Load model and prepare encryption wrapper
        torch_model = load_model()
        enc_model = EncConvNet(torch_model)

        # Homomorphic inference: apply im2col encoding, then perform forward pass
        conv_layers = [
            (torch_model.conv1.weight, torch_model.conv1.bias),
            (torch_model.conv2.weight, torch_model.conv2.bias),
            (torch_model.conv3.weight, torch_model.conv3.bias)
        ]
        enc_output = enc_model(encrypted_img_tensor, context, conv_layers)
        return render_template('result.html', output=enc_output)

    return render_template('upload.html')
