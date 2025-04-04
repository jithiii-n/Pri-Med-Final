from flask import Flask, render_template
from upload import upload_bp,  HECNN # Import the model and loader function
#from upload2 import upload2_bp, HECNN
#from upload import pipeline_bp
app = Flask(__name__)
app.register_blueprint(upload_bp, url_prefix='/upload')
#app.register_blueprint(upload2_bp, url_prefix='/upload2')
#app.register_blueprint(pipeline_bp, url_prefix='/pipeline')
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
    return render_template('upload.html', analysis_type=type)


@app.route('/pipeline', methods=['GET'])
def pipeline_visualization():
    return render_template('contact.html')
if __name__ == '__main__':
    app.run(debug=True)
