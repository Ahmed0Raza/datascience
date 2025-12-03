# Student Performance Prediction Dashboard

This project predicts student marks in Midterm I, Midterm II, and Final Examinations using regression models.

## 📋 Project Structure

```
project/
├── data/
│   ├── marks_dataset.xlsx           # Raw dataset (6 sheets)
│   └── preprocessed_dataset.csv     # Cleaned and processed dataset
├── models/                          # Trained model files
│   ├── rq1_best_model.pkl          # Model for predicting Midterm I
│   ├── rq2_best_model.pkl          # Model for predicting Midterm II
│   └── rq3_best_model.pkl          # Model for predicting Final Exam
├── outputs/
│   └── dataset_summary.txt          # Dataset statistics
├── ds_assignment.py                 # Training script (run in Colab)
├── app.py                           # Streamlit dashboard
└── requirements.txt                 # Python dependencies
```

## 🚀 Workflow

### Step 1: Train Models in Google Colab

1. Upload `ds_assignment.py` to Google Colab
2. Run all cells in the notebook
3. The script will:
   - Download dataset from Google Drive
   - Preprocess data (clean, standardize, feature engineering)
   - Train multiple regression models for each research question
   - Perform bootstrapping (500 samples) for confidence intervals
   - **Save trained models** as `.pkl` files in the `models/` folder

### Step 2: Download Trained Models

From Google Colab, download these files:
- `models/rq1_best_model.pkl`
- `models/rq2_best_model.pkl`
- `models/rq3_best_model.pkl`

### Step 3: Deploy to Streamlit

1. Create a `models/` folder in your Streamlit deployment
2. Upload the three `.pkl` files
3. Ensure `data/preprocessed_dataset.csv` is also uploaded
4. The Streamlit app will automatically load the pre-trained models

## 📊 Research Questions

### RQ1: Predicting Midterm I
- **Features:** Assignments 1-3, Quizzes 1-3, aggregate features
- **Model:** Multiple Linear Regression
- **No data leakage:** Only uses data available before Midterm I

### RQ2: Predicting Midterm II
- **Features:** All assignments, quizzes + Midterm I results
- **Model:** Multiple Linear Regression
- **No data leakage:** Does not use Midterm II or Final data

### RQ3: Predicting Final Exam
- **Features:** All assignments, quizzes + both midterms
- **Model:** Multiple Linear Regression
- **No data leakage:** Does not use Final exam data

## 🔧 Requirements

```bash
pip install -r requirements.txt
```

## 🎯 Model Details

Each `.pkl` file contains:
- Trained model object
- Feature scaler
- Feature names
- Target variable name
- Training metrics (MAE, RMSE, R²)
- Test metrics (MAE, RMSE, R²)
- Bootstrap 95% confidence interval

## 📈 Metrics

- **MAE** (Mean Absolute Error): Average prediction error
- **RMSE** (Root Mean Squared Error): Penalizes large errors  
- **R²** (R-squared): Proportion of variance explained (0-1)
- **95% CI**: Confidence interval from bootstrapping

## ✅ Key Features

- ✨ Interactive Streamlit dashboard
- 📊 Data exploration with visualizations
- 🤖 Pre-trained models (no retraining needed)
- 🎯 Real-time predictions with custom inputs
- 📈 Comprehensive model evaluation
- 🔄 Complete workflow visualization

## 🔒 No Data Leakage

The models respect temporal constraints:
- Features are selected based on what would be available at prediction time
- Train-test split before any preprocessing
- Scaler fitted only on training data
- Bootstrap uses training data only

## 📝 Running Locally (Optional)

If you have the models downloaded locally:

```bash
streamlit run app.py
```

Make sure the `models/` folder contains the three `.pkl` files!
