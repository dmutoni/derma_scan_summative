# 🩺 DermaScan — AI Dermatology Assistant  
**End-to-End Machine Learning Pipeline with Prediction, Bulk Upload & Model Retraining**

DermaScan is an AI-powered dermatology classifier that detects **11 major skin conditions** using deep learning.  
It includes a **FastAPI backend**, a **Streamlit user interface**, **bulk image upload**, **model retraining**, and **real-time predictions**.  
The full system is **live-deployed on Render** and supports scalable inference with a clean ML pipeline.

🔗 **Video Demo:** https://www.youtube.com/watch?v=toTrpcXwO0w
🔗 **Dataset Link:** https://www.kaggle.com/datasets/osmankagankurnaz/acne-dataset-in-yolov8-format
🔗 **Hosted Web Service Link** https://derma-scan-summative-1.onrender.com 

---

## 🚀 Features

### ✔️ **1. Real-time Skin Disease Prediction**
Upload a single image to receive:
- Predicted disease class  
- Confidence score  
- Automatic preprocessing  
- Support for JPG, JPEG, PNG  

### ✔️ **2. Bulk Image Upload**
Upload multiple training images at once.  
The system:
- Reads filenames  
- Infers the class from prefix  
- Automatically sorts them into folders under `data/new_data/`  

### ✔️ **3. Model Retraining**
Retrain the classifier using newly uploaded images:
- Appends new data to the existing training set  
- Retrains the CNN using 256×256 preprocessed images  
- Saves the updated model as `dermascan_retrained.h5`  
- Automatically clears `new_data/` after training  

### ✔️ **4. Full Web UI (Streamlit)**
The UI includes:
- Uptime status  
- Model class summary  
- Prediction interface  
- Bulk upload page  
- Retraining dashboard  
- Visualization section (class distribution, image samples, shape distribution)  

### ✔️ **5. FastAPI Backend**
Endpoints:
- `POST /predict` — single image inference  
- `POST /upload-bulk` — multi-image training upload  
- `POST /retrain` — retrain model  
- `GET /health` — uptime + supported classes  

### ✔️ **6. Load Testing with Locust**
Locust simulates 10+ concurrent users sending images to `/predict`:
- Measures latency  
- Ensures reliability of deployment  
- Detects bottlenecks  

### ✔️ **7. Fully Deployed on Render (Production)**  
Backend + UI hosted on Render:
- FastAPI running on port 8000  
- Automatic build & restart  
- Scalable free-tier support  

---

## 🧠 Supported Skin Disease Classes
DermaScan predicts **11 conditions**, including:

| Class |
|-------|
| acne |
| basal_cell_carcinoma |
| benign_keratosis |
| eczema |
| fungal |
| melanocytic_nevi |
| melanoma |
| normal |
| psoriasis |
| seborrheic_keratosis |
| warts |

These were trained on cleaned subsets of the ISIC dataset.

---

## Flood Test

<img width="1702" height="361" alt="image" src="https://github.com/user-attachments/assets/0f2913b6-38a5-49ff-b741-240580561f89" />

## Render Web service

<img width="1725" height="707" alt="image" src="https://github.com/user-attachments/assets/88d0318d-1674-46cf-8d7b-62a64b9212d8" />


## 📂 Project Structure
derma_scan_summative/
│
├── data/
│ ├── train/
│ ├── test/
│ └── new_data/ # uploaded bulk images
│
├── models/
│ └── dermascan_base.h5
│
├── src/
│ ├── api.py # FastAPI backend
│ ├── ui_app.py # Streamlit UI
│ ├── model.py # Model architecture & fine-tuning
│ ├── prediction.py # Preprocessing + inference
│ ├── preprocessing.py
│ └── utils.py
│
├── locustfile.py # Load testing
├── requirements.txt
├── Dockerfile
└── README.md


---

## ⚙️ Installation (Local Setup)

### **1. Clone the repo**
```bash
git clone https://github.com/dmutoni/derma_scan_summative/ 
cd derma_scan_summative
```

### **2. Install dependencies**
```pip install -r requirements.txt```

### **3. Run backend**
uvicorn src.api:app --reload

### **4. Run UI**
streamlit run src/ui_app.py

## How to Use the System
### 1. Make a Prediction

Navigate to "Predict"

Upload a skin image

Receive:

Predicted label

Confidence score

Preprocessed preview

### 2. Bulk Upload

Go to "Bulk Upload"

Upload multiple images at once

The backend:

Saves them into new_data/<class_name>/

Ensures correct naming format

### 3. Retrain Model

Go to "Retrain Model"

Click Start Retraining

The model is training using (MobileNetV2 Transfer Learning) and is fine-tuned with newly uploaded data

A fresh file dermascan_retrained.h5 is saved

🧪 Load Testing (Locust)
Run Locust:
locust -f locustfile.py


Then open:

http://localhost:8089


Simulate:

10 users

1 request per second

Continuous inference testing

## Deployment on Render
Render settings used:

Runtime: Python 3

### Build Command:

pip install -r requirements.txt


### Start Command:

uvicorn src.api:app --host 0.0.0.0 --port 8000


### Streamlit UI deployed separately using:

streamlit run src/ui_app.py --server.port $PORT --server.address 0.0.0.0


## Environment variables:

IS_RENDER = true

Author

Mutoni Denyse Uwingeneye
Machine Learning Engineering — ALU
GitHub: dmtoni
