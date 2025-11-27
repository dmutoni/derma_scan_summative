# 🩺 DermaScan — AI Dermatology Assistant  
**End-to-End Machine Learning Pipeline with Prediction, Bulk Upload & Model Retraining**

DermaScan is an AI-powered dermatology classifier that detects **11 major skin conditions** using deep learning.  
It includes a **FastAPI backend**, a **Streamlit user interface**, **bulk image upload**, **model retraining**, and **real-time predictions**.  
The full system is **live-deployed on Render** and supports scalable inference with a clean ML pipeline.

🔗 **Video Demo:** https://www.youtube.com/watch?v=toTrpcXwO0w
🔗 **Dataset Link:** https://www.kaggle.com/datasets/osmankagankurnaz/acne-dataset-in-yolov8-format

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

## 📂 Project Structure

