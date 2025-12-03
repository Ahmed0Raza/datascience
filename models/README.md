# Trained Models Directory

This folder contains the pre-trained machine learning models for student performance prediction.

## Required Files

After running `ds_assignment.py` in Google Colab, download and place these files here:

- `rq1_best_model.pkl` - Model for predicting Midterm I
- `rq2_best_model.pkl` - Model for predicting Midterm II  
- `rq3_best_model.pkl` - Model for predicting Final Exam

## File Contents

Each `.pkl` file contains a dictionary with:
- `model`: Trained sklearn LinearRegression model
- `scaler`: StandardScaler fitted on training data
- `features`: List of feature column names
- `target`: Target variable name
- `train_mae`, `train_rmse`, `train_r2`: Training metrics
- `test_mae`, `test_rmse`, `test_r2`: Test metrics
- `bootstrap_ci`: Tuple of (lower, upper) 95% confidence interval
- `model_name`: Name of the model type

## How to Generate

1. Open `ds_assignment.py` in Google Colab
2. Run all cells
3. Download the `models/` folder
4. Place the `.pkl` files in this directory
5. Deploy to Streamlit

## Note

These files are generated from your training script and should NOT be committed to Git if they're too large (>100MB). The Streamlit app will load them at runtime.
