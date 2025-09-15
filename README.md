# Crop Yield Prediction Model

This project implements a **Random Forest Regressor** wrapped in a **scikit-learn Pipeline** to predict **crop yield** using agricultural and environmental data.  
The pipeline handles **categorical encoding**, **numerical features**, and integrates preprocessing with the model for streamlined predictions.

---

## 📌 Project Overview
Crop yield prediction is an essential problem in agricultural analytics. This model helps in:
- Estimating yield based on historical and environmental conditions.  
- Supporting farmers in making informed decisions.  
- Providing policymakers with reliable insights for food supply planning.  

The model was trained using a **Kaggle dataset** containing crop, weather, and soil data.  

---

## 📊 Dataset
- Source: [Kaggle – Crop Yield Dataset](https://www.kaggle.com/)  
- Features include:
  - **Categorical:** `Crop`, `Season`, `State`  
  - **Numerical:** `Crop_Year`, `Area`, `Production`, `Annual_Rainfall`, `Fertilizer`, `Pesticide`  
- Target variable: **Yield (kg/hectare)**  

⚠️ Note: In this implementation, extreme yield values are clipped at an upper limit of **20** for stability.

---

## 🧠 Model
The pipeline consists of:
1. **Preprocessing**  
   - OneHotEncoding for categorical features.  
   - Passthrough for numerical features.  

2. **Random Forest Regressor**  
   - `n_estimators=200`  
   - `random_state=42`  

### Evaluation Metrics
- **Mean Absolute Error (MAE)**  
- **Mean Squared Error (MSE)**  
- **R² Score**  

---

## ⚙️ Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/crop-yield-prediction.git
cd crop-yield-prediction
pip install -r requirements.txt
