from flask import Flask, render_template
from upload import upload_bp
from upload2 import upload2_bp

app = Flask(__name__)

# Register blueprints with proper URL prefixes
app.register_blueprint(upload_bp, url_prefix='/upload')
app.register_blueprint(upload2_bp, url_prefix='/upload2')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/upload/<type>')
def upload_page(type):
    # Validate the type parameter
    if type not in ['chest', 'alzheimer']:
        return "Invalid analysis type", 400
    return render_template('upload.html', type=type)

@app.route('/pipeline')
def pipeline_visualization():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)