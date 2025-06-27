
# 🐔 Poultry Disease Classification Using Transfer Learning

This project uses **Transfer Learning** with pre-trained models (VGG16, VGG19, ResNet50) to classify poultry diseases into four categories:

- 🦠 **Coccidiosis**
- 🧬 **Salmonella**
- 🦠 **New Castle Disease**
- ✅ **Healthy**

The trained model is deployed in a **Flask web application**, allowing users (farmers, veterinarians, students) to upload poultry images and receive instant diagnosis.

---

## 🚀 Features

- ✅ Image classification using VGG16 / VGG19 / ResNet50
- 📦 Lightweight model with high accuracy
- 🌐 Flask-based web application interface
- 🧠 Trained on real-world poultry fecal image dataset
- 📊 Real-time predictions on user-uploaded images

---

## 🗃️ Dataset

We used a small, well-labeled dataset of ~1,200 poultry fecal images from:

🔗 **[Zenodo - PCR-annotated poultry disease images](https://zenodo.org/records/5801834)**

Dataset classes:
- `Coccidiosis`
- `Healthy`
- `Salmonella`
- `New Castle Disease`

---

## 🧰 Tech Stack

- 🐍 Python 3.x
- 🧠 TensorFlow / Keras
- 🔍 OpenCV
- 🌐 Flask
- 📦 Numpy, Pandas, Scikit-learn, Matplotlib

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/poultry-disease-classification.git
cd poultry-disease-classification
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
# Activate it:
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> Make sure Python is added to your system PATH before running.

---

## 📁 Project Structure

```bash
├── app.py                   # Flask app entry point
├── train_model.py           # Training script using VGG16
├── model/
│   └── poultry_model.h5     # Trained Keras model
├── static/
│   └── uploads/             # Uploaded images
├── templates/
│   └── index.html           # HTML frontend
├── dataset/                 # Organized training/validation/test data
└── README.md
```

---

## 🧪 Usage

### To train the model

```bash
python train_model.py
```

### To run the Flask app

```bash
python app.py
```

Then open your browser and visit:  
[http://127.0.0.1:5000](http://127.0.0.1:5000)



## 🧑‍💻 Author

- Devalla Dinesh  

Project done under the **SmartInternz Virtual Internship Program 2025** at RISE Krishna Sai Gandhi Group of Institutions.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
