import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
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

# Load pre-trained model
@st.cache_resource
def load_model(rq_name):
    """Load pre-trained model for a research question"""
    try:
        model_path = f'models/{rq_name}_best_model.pkl'
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data, True
    except FileNotFoundError:
        st.warning(f"⚠️ Pre-trained model not found: {model_path}")
        st.info("Please run ds_assignment.py in Google Colab and download the models folder.")
        return None, False

def plot_predictions(y_test, y_pred, title):
    """Create scatter plot of predictions vs actual"""
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    
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
    │  2. Multiple Linear Regression ✅ (BEST - SAVED)                │
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
    └────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                   SAVE BEST MODEL AS .PKL                       │
    │         (Download from Colab and upload to Streamlit)           │
    └─────────────────────────────────────────────────────────────────┘
    ```
    """
    st.code(workflow, language='')

def display_rq_results(rq_name, rq_title, target_description):
    """Display results for a research question"""
    st.markdown(f"## 📈 {rq_title}")
    st.info(f"**Objective:** {target_description}")
    
    # Load pre-trained model
    model_data, loaded = load_model(rq_name)
    
    if not loaded:
        st.error(f"""
        ❌ **Model not found!**
        
        To load the model:
        1. Run `ds_assignment.py` in Google Colab
        2. Download the generated `models/{rq_name}_best_model.pkl` file
        3. Upload it to your Streamlit deployment in the `models/` folder
        """)
        return
    
    # Display model information
    st.success(f"✅ Loaded pre-trained model: **{model_data['model_name']}**")
    
    # Model Performance Metrics
    st.markdown("### 📊 Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Test MAE", f"{model_data['test_mae']:.3f}")
        st.metric("Train MAE", f"{model_data['train_mae']:.3f}")
    
    with col2:
        st.metric("Test RMSE", f"{model_data['test_rmse']:.3f}")
        st.metric("Train RMSE", f"{model_data['train_rmse']:.3f}")
    
    with col3:
        st.metric("Test R²", f"{model_data['test_r2']:.3f}")
        st.metric("Train R²", f"{model_data['train_r2']:.3f}")
    
    # Bootstrap CI
    st.markdown("### 🔄 Bootstrap Confidence Interval (95%)")
    ci_lower, ci_upper = model_data['bootstrap_ci']
    st.info(f"**MAE 95% CI:** [{ci_lower:.3f}, {ci_upper:.3f}] (Width: {ci_upper - ci_lower:.3f})")
    
    with st.expander("💡 Interpretation"):
        st.markdown("""
        The 95% confidence interval shows the range in which we can be 95% confident that the true MAE lies.
        - **Narrower CI** = More stable and reliable model
        - Bootstrap uses **training data only** to avoid data leakage
        """)
    
    # Overfitting Analysis
    st.markdown("### 🔍 Overfitting Analysis")
    train_test_diff = abs(model_data['train_r2'] - model_data['test_r2'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        if train_test_diff < 0.1:
            st.success(f"✅ **Good Fit**\n\nR² difference: {train_test_diff:.3f}")
            st.write("The model generalizes well to unseen data.")
        elif train_test_diff < 0.2:
            st.warning(f"⚠️ **Slight Overfitting**\n\nR² difference: {train_test_diff:.3f}")
            st.write("The model performs slightly better on training data.")
        else:
            st.error(f"❌ **Overfitting Detected**\n\nR² difference: {train_test_diff:.3f}")
            st.write("The model is overfitting to the training data.")
    
    with col2:
        # Feature importance
        st.markdown("**Model Features:**")
        st.write(f"- Target: `{model_data['target']}`")
        st.write(f"- Features used: {len(model_data['features'])}")
        with st.expander("View all features"):
            for feat in model_data['features']:
                st.write(f"  • {feat}")
    
    # Prediction Demo
    st.markdown("### 🎯 Make a Prediction")
    
    with st.expander("Try the model with custom inputs"):
        st.write(f"Enter values for the {len(model_data['features'])} features:")
        
        # Create input fields for each feature
        user_inputs = {}
        col_count = 3
        cols = st.columns(col_count)
        
        for idx, feature in enumerate(model_data['features']):
            with cols[idx % col_count]:
                user_inputs[feature] = st.number_input(
                    feature,
                    min_value=0.0,
                    max_value=200.0,
                    value=50.0,
                    step=1.0,
                    key=f"{rq_name}_{feature}"
                )
        
        if st.button(f"Predict {model_data['target']}", key=f"predict_{rq_name}"):
            # Prepare input data
            input_df = pd.DataFrame([user_inputs])
            
            # Scale the input
            input_scaled = model_data['scaler'].transform(input_df)
            
            # Make prediction
            prediction = model_data['model'].predict(input_scaled)[0]
            
            st.success(f"### Predicted {model_data['target']}: **{prediction:.2f}**")

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
    
    # Check if models exist
    models_exist = all([
        os.path.exists('models/rq1_best_model.pkl'),
        os.path.exists('models/rq2_best_model.pkl'),
        os.path.exists('models/rq3_best_model.pkl')
    ])
    
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
        
        st.markdown("---")
        st.markdown("### 🤖 Model Status")
        if models_exist:
            st.success("✅ All models loaded")
        else:
            st.error("❌ Models not found")
            st.caption("Run ds_assignment.py in Colab")
    
    # Overview Page
    if page == "🏠 Overview":
        st.markdown("## 🎯 Research Questions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>RQ1</h3>
                <p><strong>How accurately can we predict student marks in Midterm I?</strong></p>
                <p style="font-size: 0.9rem; color: #000;">Using assignments and quizzes completed before Midterm I</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>RQ2</h3>
                <p><strong>How accurately can we predict student marks in Midterm II?</strong></p>
                <p style="font-size: 0.9rem; color: #000;">Using assignments, quizzes, and Midterm I results</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>RQ3</h3>
                <p><strong>How accurately can we predict final examination marks?</strong></p>
                <p style="font-size: 0.9rem; color: #000;">Using all assignments, quizzes, and midterm results</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("## 📋 Project Overview")
        
        st.markdown("""
        This dashboard presents the results of a comprehensive student performance prediction analysis. 
        The project involves:
        
        - **Data Source:** 6 Excel sheets containing student assessment scores (272 students total)
        - **Assessments:** Assignments, Quizzes, Midterms, and Final Exams
        - **Model:** Multiple Linear Regression (best performing model)
        - **Evaluation Metrics:** MAE, RMSE, R² (with 95% confidence intervals via bootstrapping)
        - **Validation:** Train/Test split to assess overfitting/underfitting
        
        Navigate through the sidebar to explore predictions for each research question!
        """)
        
        if not models_exist:
            st.warning("""
            ### ⚠️ Models Not Found
            
            To use this dashboard:
            1. Open and run `ds_assignment.py` in **Google Colab**
            2. The script will train models and save them as `.pkl` files
            3. Download the `models/` folder from Colab
            4. Upload the `.pkl` files to your Streamlit deployment
            
            Expected files:
            - `models/rq1_best_model.pkl`
            - `models/rq2_best_model.pkl`
            - `models/rq3_best_model.pkl`
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
        display_rq_results(
            "rq1",
            "RQ1: Predicting Midterm I Marks",
            "Predict Midterm I scores using assignments and quizzes completed BEFORE Midterm I"
        )
    
    # RQ2: Midterm II Prediction
    elif page == "📈 RQ2: Midterm II":
        display_rq_results(
            "rq2",
            "RQ2: Predicting Midterm II Marks",
            "Predict Midterm II scores using assignments, quizzes, and Midterm I results"
        )
    
    # RQ3: Final Exam Prediction
    elif page == "📈 RQ3: Final Exam":
        display_rq_results(
            "rq3",
            "RQ3: Predicting Final Examination Marks",
            "Predict Final Exam scores using all assignments, quizzes, and midterm results"
        )
    
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
            **Multiple Linear Regression** was selected as the best model:
            
            - Uses all available features for prediction
            - Balances performance and simplicity
            - Bootstrapping (500 samples) for confidence intervals
            - Scaled features to ensure fair comparison
            """)
        
        with st.expander("5️⃣ Model Evaluation"):
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