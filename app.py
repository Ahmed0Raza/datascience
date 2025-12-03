import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Student Performance Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_data():
    """Load the preprocessed dataset"""
    try:
        df = pd.read_csv('data/preprocessed_dataset.csv')
        return df
    except FileNotFoundError:
        st.error("⚠️ Preprocessed dataset not found! Please ensure 'data/preprocessed_dataset.csv' exists.")
        return None

# Cache model training
@st.cache_data
def train_models(df, target_col, feature_cols):
    """Train multiple regression models for a given target"""
    # Prepare data
    X = df[feature_cols].fillna(0)
    y = df[target_col].fillna(0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {}
    
    # 1. Simple Linear Regression (using first feature only)
    simple_model = LinearRegression()
    simple_model.fit(X_train[[feature_cols[0]]], y_train)
    simple_pred_train = simple_model.predict(X_train[[feature_cols[0]]])
    simple_pred_test = simple_model.predict(X_test[[feature_cols[0]]])
    
    results['Simple Linear'] = {
        'model': simple_model,
        'train_mae': mean_absolute_error(y_train, simple_pred_train),
        'test_mae': mean_absolute_error(y_test, simple_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, simple_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, simple_pred_test)),
        'train_r2': r2_score(y_train, simple_pred_train),
        'test_r2': r2_score(y_test, simple_pred_test),
        'y_test': y_test,
        'y_pred': simple_pred_test
    }
    
    # 2. Multiple Linear Regression
    multiple_model = LinearRegression()
    multiple_model.fit(X_train, y_train)
    multiple_pred_train = multiple_model.predict(X_train)
    multiple_pred_test = multiple_model.predict(X_test)
    
    results['Multiple Linear'] = {
        'model': multiple_model,
        'train_mae': mean_absolute_error(y_train, multiple_pred_train),
        'test_mae': mean_absolute_error(y_test, multiple_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, multiple_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, multiple_pred_test)),
        'train_r2': r2_score(y_train, multiple_pred_train),
        'test_r2': r2_score(y_test, multiple_pred_test),
        'y_test': y_test,
        'y_pred': multiple_pred_test
    }
    
    # 3. Polynomial Regression (degree 2)
    poly_features = PolynomialFeatures(degree=2)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    poly_pred_train = poly_model.predict(X_train_poly)
    poly_pred_test = poly_model.predict(X_test_poly)
    
    results['Polynomial (deg=2)'] = {
        'model': poly_model,
        'train_mae': mean_absolute_error(y_train, poly_pred_train),
        'test_mae': mean_absolute_error(y_test, poly_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, poly_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, poly_pred_test)),
        'train_r2': r2_score(y_train, poly_pred_train),
        'test_r2': r2_score(y_test, poly_pred_test),
        'y_test': y_test,
        'y_pred': poly_pred_test
    }
    
    # 4. Dummy Regressor (Baseline)
    dummy_model = DummyRegressor(strategy='mean')
    dummy_model.fit(X_train, y_train)
    dummy_pred_train = dummy_model.predict(X_train)
    dummy_pred_test = dummy_model.predict(X_test)
    
    results['Dummy (Baseline)'] = {
        'model': dummy_model,
        'train_mae': mean_absolute_error(y_train, dummy_pred_train),
        'test_mae': mean_absolute_error(y_test, dummy_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, dummy_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, dummy_pred_test)),
        'train_r2': r2_score(y_train, dummy_pred_train),
        'test_r2': r2_score(y_test, dummy_pred_test),
        'y_test': y_test,
        'y_pred': dummy_pred_test
    }
    
    # Bootstrap confidence intervals (500 samples) - using training data only
    bootstrap_results = {}
    n_bootstrap = 500
    
    for model_name in ['Simple Linear', 'Multiple Linear', 'Polynomial (deg=2)']:
        mae_scores = []
        for _ in range(n_bootstrap):
            # Resample training data
            indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
            X_boot = X_train.iloc[indices]
            y_boot = y_train.iloc[indices]
            
            if model_name == 'Simple Linear':
                temp_model = LinearRegression()
                temp_model.fit(X_boot[[feature_cols[0]]], y_boot)
                pred = temp_model.predict(X_boot[[feature_cols[0]]])
            elif model_name == 'Multiple Linear':
                temp_model = LinearRegression()
                temp_model.fit(X_boot, y_boot)
                pred = temp_model.predict(X_boot)
            else:  # Polynomial
                X_boot_poly = poly_features.transform(X_boot)
                temp_model = LinearRegression()
                temp_model.fit(X_boot_poly, y_boot)
                pred = temp_model.predict(X_boot_poly)
            
            mae_scores.append(mean_absolute_error(y_boot, pred))
        
        # Calculate 95% confidence interval
        ci_lower = np.percentile(mae_scores, 2.5)
        ci_upper = np.percentile(mae_scores, 97.5)
        bootstrap_results[model_name] = {
            'mae_mean': np.mean(mae_scores),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    return results, bootstrap_results

def plot_predictions(y_test, y_pred, title):
    """Create scatter plot of predictions vs actual"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', s=50)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_residuals(y_test, y_pred, title):
    """Create residual plot"""
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', s=50)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    
    ax.set_xlabel('Predicted Values', fontsize=12)
    ax.set_ylabel('Residuals', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    return fig

def display_workflow():
    """Display the workflow diagram"""
    st.markdown("### 📋 Data Processing & Model Training Workflow")
    
    workflow = """
    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                     RAW DATA (6 Excel Sheets)                   │
    └────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │               DATA CLEANING & PREPROCESSING                     │
    │  • Remove non-student rows (Weightage, Total)                   │
    │  • Handle missing values (fill with 0)                          │
    │  • Standardize column names                                     │
    │  • Create engineered features (mean, max, min, std, CV)         │
    └────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │               COMBINE ALL SHEETS INTO ONE DATASET               │
    │                  (272 students, 45 features)                    │
    └────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                     FEATURE SELECTION                           │
    │  RQ1 (Midterm I): Use Assignments & Quizzes before Midterm I   │
    │  RQ2 (Midterm II): Use Assignments, Quizzes & Midterm I        │
    │  RQ3 (Final): Use all Assignments, Quizzes & Midterms          │
    └────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                   TRAIN-TEST SPLIT (80-20)                      │
    └────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                      MODEL TRAINING                             │
    │  1. Simple Linear Regression                                    │
    │  2. Multiple Linear Regression                                  │
    │  3. Polynomial Regression (degree=2)                            │
    │  4. Dummy Regressor (Baseline)                                  │
    └────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │              BOOTSTRAPPING (500 samples, train only)            │
    │                Calculate 95% CI for MAE                         │
    └────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    MODEL EVALUATION                             │
    │              MAE, RMSE, R² (Train & Test)                       │
    └─────────────────────────────────────────────────────────────────┘
    ```
    """
    st.code(workflow, language='')

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Student Performance Prediction Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Predicting Student Marks using Regression Models**")
    st.markdown("---")
    
    # Load data
    df = load_data()
    
    if df is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/000000/student-male.png", width=150)
        st.title("Navigation")
        page = st.radio(
            "Select Page:",
            ["🏠 Overview", "📈 RQ1: Midterm I", "📈 RQ2: Midterm II", "📈 RQ3: Final Exam", "📊 Data Exploration", "🔄 Workflow"]
        )
        
        st.markdown("---")
        st.markdown("### 📚 Dataset Info")
        st.info(f"**Total Students:** {len(df)}")
        st.info(f"**Total Features:** {df.shape[1]}")
    
    # Overview Page
    if page == "🏠 Overview":
        st.markdown("## 🎯 Research Questions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>RQ1</h3>
                <p><strong>How accurately can we predict student marks in Midterm I?</strong></p>
                <p style="font-size: 0.9rem; color: #555;">Using assignments and quizzes completed before Midterm I</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>RQ2</h3>
                <p><strong>How accurately can we predict student marks in Midterm II?</strong></p>
                <p style="font-size: 0.9rem; color: #555;">Using assignments, quizzes, and Midterm I results</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>RQ3</h3>
                <p><strong>How accurately can we predict final examination marks?</strong></p>
                <p style="font-size: 0.9rem; color: #555;">Using all assignments, quizzes, and midterm results</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("## 📋 Project Overview")
        
        st.markdown("""
        This dashboard presents the results of a comprehensive student performance prediction analysis. 
        The project involves:
        
        - **Data Source:** 6 Excel sheets containing student assessment scores (272 students total)
        - **Assessments:** Assignments, Quizzes, Midterms, and Final Exams
        - **Models:** Simple Linear, Multiple Linear, Polynomial Regression, and Baseline (Dummy)
        - **Evaluation Metrics:** MAE, RMSE, R² (with 95% confidence intervals via bootstrapping)
        - **Validation:** Train/Test split to assess overfitting/underfitting
        
        Navigate through the sidebar to explore predictions for each research question!
        """)
        
        st.markdown("---")
        st.markdown("## 📊 Quick Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Students", len(df))
        with col2:
            st.metric("Total Features", df.shape[1])
        with col3:
            st.metric("Assignments", len([c for c in df.columns if 'Assignment' in c and '_' in c and 'Mean' not in c]))
        with col4:
            st.metric("Quizzes", len([c for c in df.columns if 'Quiz' in c and '_' in c and 'Mean' not in c]))
    
    # RQ1: Midterm I Prediction
    elif page == "📈 RQ1: Midterm I":
        st.markdown("## 📈 RQ1: Predicting Midterm I Marks")
        st.info("**Objective:** Predict Midterm I scores using assignments and quizzes completed BEFORE Midterm I")
        
        # Select features (assignments and quizzes before Midterm I)
        feature_cols = ['Assignment_1', 'Assignment_2', 'Assignment_3', 'Quiz_1', 'Quiz_2', 'Quiz_3', 
                       'Assignment_Mean', 'Quiz_Mean']
        target_col = 'Midterm_1'
        
        # Train models
        with st.spinner("Training models..."):
            results, bootstrap_results = train_models(df, target_col, feature_cols)
        
        # Display results table
        st.markdown("### 📊 Model Comparison Table")
        
        comparison_data = []
        for model_name, metrics in results.items():
            comparison_data.append({
                'Model': model_name,
                'Train MAE': f"{metrics['train_mae']:.3f}",
                'Test MAE': f"{metrics['test_mae']:.3f}",
                'Train RMSE': f"{metrics['train_rmse']:.3f}",
                'Test RMSE': f"{metrics['test_rmse']:.3f}",
                'Train R²': f"{metrics['train_r2']:.3f}",
                'Test R²': f"{metrics['test_r2']:.3f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Bootstrap results
        st.markdown("### 🔄 Bootstrap Results (95% Confidence Interval for MAE)")
        st.markdown("*Based on 500 bootstrap samples from training data*")
        
        bootstrap_data = []
        for model_name, boot_metrics in bootstrap_results.items():
            bootstrap_data.append({
                'Model': model_name,
                'Mean MAE': f"{boot_metrics['mae_mean']:.3f}",
                'CI Lower (2.5%)': f"{boot_metrics['ci_lower']:.3f}",
                'CI Upper (97.5%)': f"{boot_metrics['ci_upper']:.3f}",
                'CI Width': f"{boot_metrics['ci_upper'] - boot_metrics['ci_lower']:.3f}"
            })
        
        bootstrap_df = pd.DataFrame(bootstrap_data)
        st.dataframe(bootstrap_df, use_container_width=True)
        
        # Interpretation
        with st.expander("💡 Interpretation of Bootstrap Results"):
            st.markdown("""
            The 95% confidence intervals show the range in which we can be 95% confident that the true MAE lies.
            
            - **Narrower CI** = More stable and reliable model
            - **CI not overlapping with baseline** = Model performs significantly better than random guessing
            - The bootstrap uses **training data only** to avoid data leakage
            """)
        
        # Best model visualization
        st.markdown("### 🏆 Best Model Performance")
        
        # Find best model (lowest test MAE, excluding dummy)
        best_model_name = min(
            [k for k in results.keys() if k != 'Dummy (Baseline)'],
            key=lambda k: results[k]['test_mae']
        )
        best_model = results[best_model_name]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### {best_model_name}")
            st.metric("Test MAE", f"{best_model['test_mae']:.3f}")
            st.metric("Test RMSE", f"{best_model['test_rmse']:.3f}")
            st.metric("Test R²", f"{best_model['test_r2']:.3f}")
            
            # Overfitting check
            st.markdown("**Overfitting Analysis:**")
            train_test_diff = abs(best_model['train_r2'] - best_model['test_r2'])
            if train_test_diff < 0.1:
                st.success(f"✅ Good fit (R² difference: {train_test_diff:.3f})")
            elif train_test_diff < 0.2:
                st.warning(f"⚠️ Slight overfitting (R² difference: {train_test_diff:.3f})")
            else:
                st.error(f"❌ Overfitting detected (R² difference: {train_test_diff:.3f})")
        
        with col2:
            fig = plot_predictions(best_model['y_test'], best_model['y_pred'], 
                                 f"Predictions vs Actual - {best_model_name}")
            st.pyplot(fig)
        
        # Residual plot
        st.markdown("### 📉 Residual Analysis")
        fig_residuals = plot_residuals(best_model['y_test'], best_model['y_pred'],
                                      f"Residual Plot - {best_model_name}")
        st.pyplot(fig_residuals)
    
    # RQ2: Midterm II Prediction
    elif page == "📈 RQ2: Midterm II":
        st.markdown("## 📈 RQ2: Predicting Midterm II Marks")
        st.info("**Objective:** Predict Midterm II scores using assignments, quizzes, and Midterm I results")
        
        # Select features (everything before Midterm II)
        feature_cols = ['Assignment_1', 'Assignment_2', 'Assignment_3', 'Assignment_4',
                       'Quiz_1', 'Quiz_2', 'Quiz_3', 'Quiz_4', 'Quiz_5',
                       'Midterm_1', 'Assignment_Mean', 'Quiz_Mean']
        target_col = 'Midterm_2'
        
        # Train models
        with st.spinner("Training models..."):
            results, bootstrap_results = train_models(df, target_col, feature_cols)
        
        # Display results table
        st.markdown("### 📊 Model Comparison Table")
        
        comparison_data = []
        for model_name, metrics in results.items():
            comparison_data.append({
                'Model': model_name,
                'Train MAE': f"{metrics['train_mae']:.3f}",
                'Test MAE': f"{metrics['test_mae']:.3f}",
                'Train RMSE': f"{metrics['train_rmse']:.3f}",
                'Test RMSE': f"{metrics['test_rmse']:.3f}",
                'Train R²': f"{metrics['train_r2']:.3f}",
                'Test R²': f"{metrics['test_r2']:.3f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Bootstrap results
        st.markdown("### 🔄 Bootstrap Results (95% Confidence Interval for MAE)")
        st.markdown("*Based on 500 bootstrap samples from training data*")
        
        bootstrap_data = []
        for model_name, boot_metrics in bootstrap_results.items():
            bootstrap_data.append({
                'Model': model_name,
                'Mean MAE': f"{boot_metrics['mae_mean']:.3f}",
                'CI Lower (2.5%)': f"{boot_metrics['ci_lower']:.3f}",
                'CI Upper (97.5%)': f"{boot_metrics['ci_upper']:.3f}",
                'CI Width': f"{boot_metrics['ci_upper'] - boot_metrics['ci_lower']:.3f}"
            })
        
        bootstrap_df = pd.DataFrame(bootstrap_data)
        st.dataframe(bootstrap_df, use_container_width=True)
        
        # Interpretation
        with st.expander("💡 Interpretation of Bootstrap Results"):
            st.markdown("""
            The 95% confidence intervals show the range in which we can be 95% confident that the true MAE lies.
            
            - **Narrower CI** = More stable and reliable model
            - **CI not overlapping with baseline** = Model performs significantly better than random guessing
            - The bootstrap uses **training data only** to avoid data leakage
            """)
        
        # Best model visualization
        st.markdown("### 🏆 Best Model Performance")
        
        # Find best model
        best_model_name = min(
            [k for k in results.keys() if k != 'Dummy (Baseline)'],
            key=lambda k: results[k]['test_mae']
        )
        best_model = results[best_model_name]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### {best_model_name}")
            st.metric("Test MAE", f"{best_model['test_mae']:.3f}")
            st.metric("Test RMSE", f"{best_model['test_rmse']:.3f}")
            st.metric("Test R²", f"{best_model['test_r2']:.3f}")
            
            # Overfitting check
            st.markdown("**Overfitting Analysis:**")
            train_test_diff = abs(best_model['train_r2'] - best_model['test_r2'])
            if train_test_diff < 0.1:
                st.success(f"✅ Good fit (R² difference: {train_test_diff:.3f})")
            elif train_test_diff < 0.2:
                st.warning(f"⚠️ Slight overfitting (R² difference: {train_test_diff:.3f})")
            else:
                st.error(f"❌ Overfitting detected (R² difference: {train_test_diff:.3f})")
        
        with col2:
            fig = plot_predictions(best_model['y_test'], best_model['y_pred'], 
                                 f"Predictions vs Actual - {best_model_name}")
            st.pyplot(fig)
        
        # Residual plot
        st.markdown("### 📉 Residual Analysis")
        fig_residuals = plot_residuals(best_model['y_test'], best_model['y_pred'],
                                      f"Residual Plot - {best_model_name}")
        st.pyplot(fig_residuals)
    
    # RQ3: Final Exam Prediction
    elif page == "📈 RQ3: Final Exam":
        st.markdown("## 📈 RQ3: Predicting Final Examination Marks")
        st.info("**Objective:** Predict Final Exam scores using all assignments, quizzes, and midterm results")
        
        # Select features (everything before Final)
        feature_cols = ['Assignment_1', 'Assignment_2', 'Assignment_3', 'Assignment_4',
                       'Quiz_1', 'Quiz_2', 'Quiz_3', 'Quiz_4', 'Quiz_5', 'Quiz_6',
                       'Midterm_1', 'Midterm_2', 'Assignment_Mean', 'Quiz_Mean']
        target_col = 'Final_1'
        
        # Train models
        with st.spinner("Training models..."):
            results, bootstrap_results = train_models(df, target_col, feature_cols)
        
        # Display results table
        st.markdown("### 📊 Model Comparison Table")
        
        comparison_data = []
        for model_name, metrics in results.items():
            comparison_data.append({
                'Model': model_name,
                'Train MAE': f"{metrics['train_mae']:.3f}",
                'Test MAE': f"{metrics['test_mae']:.3f}",
                'Train RMSE': f"{metrics['train_rmse']:.3f}",
                'Test RMSE': f"{metrics['test_rmse']:.3f}",
                'Train R²': f"{metrics['train_r2']:.3f}",
                'Test R²': f"{metrics['test_r2']:.3f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Bootstrap results
        st.markdown("### 🔄 Bootstrap Results (95% Confidence Interval for MAE)")
        st.markdown("*Based on 500 bootstrap samples from training data*")
        
        bootstrap_data = []
        for model_name, boot_metrics in bootstrap_results.items():
            bootstrap_data.append({
                'Model': model_name,
                'Mean MAE': f"{boot_metrics['mae_mean']:.3f}",
                'CI Lower (2.5%)': f"{boot_metrics['ci_lower']:.3f}",
                'CI Upper (97.5%)': f"{boot_metrics['ci_upper']:.3f}",
                'CI Width': f"{boot_metrics['ci_upper'] - boot_metrics['ci_lower']:.3f}"
            })
        
        bootstrap_df = pd.DataFrame(bootstrap_data)
        st.dataframe(bootstrap_df, use_container_width=True)
        
        # Interpretation
        with st.expander("💡 Interpretation of Bootstrap Results"):
            st.markdown("""
            The 95% confidence intervals show the range in which we can be 95% confident that the true MAE lies.
            
            - **Narrower CI** = More stable and reliable model
            - **CI not overlapping with baseline** = Model performs significantly better than random guessing
            - The bootstrap uses **training data only** to avoid data leakage
            """)
        
        # Best model visualization
        st.markdown("### 🏆 Best Model Performance")
        
        # Find best model
        best_model_name = min(
            [k for k in results.keys() if k != 'Dummy (Baseline)'],
            key=lambda k: results[k]['test_mae']
        )
        best_model = results[best_model_name]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### {best_model_name}")
            st.metric("Test MAE", f"{best_model['test_mae']:.3f}")
            st.metric("Test RMSE", f"{best_model['test_rmse']:.3f}")
            st.metric("Test R²", f"{best_model['test_r2']:.3f}")
            
            # Overfitting check
            st.markdown("**Overfitting Analysis:**")
            train_test_diff = abs(best_model['train_r2'] - best_model['test_r2'])
            if train_test_diff < 0.1:
                st.success(f"✅ Good fit (R² difference: {train_test_diff:.3f})")
            elif train_test_diff < 0.2:
                st.warning(f"⚠️ Slight overfitting (R² difference: {train_test_diff:.3f})")
            else:
                st.error(f"❌ Overfitting detected (R² difference: {train_test_diff:.3f})")
        
        with col2:
            fig = plot_predictions(best_model['y_test'], best_model['y_pred'], 
                                 f"Predictions vs Actual - {best_model_name}")
            st.pyplot(fig)
        
        # Residual plot
        st.markdown("### 📉 Residual Analysis")
        fig_residuals = plot_residuals(best_model['y_test'], best_model['y_pred'],
                                      f"Residual Plot - {best_model_name}")
        st.pyplot(fig_residuals)
    
    # Data Exploration Page
    elif page == "📊 Data Exploration":
        st.markdown("## 📊 Data Exploration")
        
        st.markdown("### 📋 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("### 📈 Statistical Summary")
        st.dataframe(df.describe().T, use_container_width=True)
        
        st.markdown("### 📉 Feature Distributions")
        
        # Select feature to visualize
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_feature = st.selectbox("Select a feature to visualize:", numeric_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df[selected_feature].dropna(), bins=30, edgecolor='black', alpha=0.7)
            ax.set_xlabel(selected_feature, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'Distribution of {selected_feature}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.boxplot(df[selected_feature].dropna(), vert=True)
            ax.set_ylabel(selected_feature, fontsize=12)
            ax.set_title(f'Box Plot of {selected_feature}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        st.markdown("### 🔗 Correlation Heatmap")
        
        # Select subset of features for correlation
        correlation_features = st.multiselect(
            "Select features for correlation analysis:",
            numeric_cols,
            default=['Assignment_Mean', 'Quiz_Mean', 'Midterm_1', 'Midterm_2', 'Final_1'][:min(5, len(numeric_cols))]
        )
        
        if len(correlation_features) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = df[correlation_features].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
            ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
            st.pyplot(fig)
        else:
            st.warning("Please select at least 2 features for correlation analysis")
    
    # Workflow Page
    elif page == "🔄 Workflow":
        display_workflow()
        
        st.markdown("---")
        st.markdown("### 📝 Key Steps Explanation")
        
        with st.expander("1️⃣ Data Preprocessing"):
            st.markdown("""
            - Removed non-student rows (e.g., Weightage, Total)
            - Handled missing values by filling with 0 (assumes no submission)
            - Standardized column names across all 6 sheets
            - Created engineered features: mean, max, min, standard deviation, coefficient of variation
            """)
        
        with st.expander("2️⃣ Feature Selection (Domain Knowledge)"):
            st.markdown("""
            **Critical:** We ensure no data leakage by respecting temporal constraints:
            
            - **RQ1 (Midterm I):** Only use assignments and quizzes completed BEFORE Midterm I
            - **RQ2 (Midterm II):** Can use Midterm I results, but NOT Midterm II or Final
            - **RQ3 (Final):** Can use all midterm results, but NOT Final exam scores
            """)
        
        with st.expander("3️⃣ Train-Test Split"):
            st.markdown("""
            - **80% Training Data:** Used to fit the models
            - **20% Test Data:** Used to evaluate model performance on unseen data
            - **Random State = 42:** Ensures reproducibility
            """)
        
        with st.expander("4️⃣ Model Training"):
            st.markdown("""
            **Four Models Trained:**
            
            1. **Simple Linear Regression:** Uses only one predictor variable
            2. **Multiple Linear Regression:** Uses all selected features
            3. **Polynomial Regression (degree=2):** Captures non-linear relationships
            4. **Dummy Regressor (Baseline):** Predicts mean value (random guessing)
            """)
        
        with st.expander("5️⃣ Bootstrapping"):
            st.markdown("""
            - **500 bootstrap samples** from training data only
            - Calculates **95% confidence interval** for MAE
            - Provides uncertainty quantification for model performance
            - **No data leakage:** Uses only training data for bootstrapping
            """)
        
        with st.expander("6️⃣ Model Evaluation"):
            st.markdown("""
            **Metrics Used:**
            
            - **MAE (Mean Absolute Error):** Average absolute difference between predictions and actual values
            - **RMSE (Root Mean Squared Error):** Square root of average squared differences (penalizes large errors)
            - **R² (Coefficient of Determination):** Proportion of variance explained by the model (0-1, higher is better)
            
            **Overfitting Detection:**
            - Compare Train R² vs Test R²
            - Large difference indicates overfitting
            """)

if __name__ == "__main__":
    main()