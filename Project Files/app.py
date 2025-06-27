import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
MODEL = load_model("model/poultry_model.h5")
UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

classes = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No file part"
    file = request.files['image']
    if file.filename == '':
        return "No image selected"

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    image = load_img(filepath, target_size=(224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = MODEL.predict(image)
    label = classes[np.argmax(prediction)]

    return render_template('index.html', prediction=label, img_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
