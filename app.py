from flask import Flask, render_template

from upload import upload_bp

app = Flask(__name__)
app.register_blueprint(upload_bp, url_prefix='/upload')
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
