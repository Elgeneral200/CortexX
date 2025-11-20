"""
CortexX - Enterprise Sales Forecasting Platform
================================================
Complete Professional Dashboard - Milestones 1-4

Version: 2.0 Professional Edition 
Date: November 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
import logging
from typing import Optional, Dict, Any

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# Import real modules
try:
    from src.data.collection import DataCollector
    from src.data.preprocessing import DataPreprocessor
    from src.data.exploration import DataExplorer
    from src.features.engineering import FeatureEngineer
    from src.features.selection import FeatureSelector
    from src.models.training import ModelTrainer
    from src.models.evaluation import ModelEvaluator
    from src.models.deployment import ModelDeployer
    from src.models.optimization import HyperparameterOptimizer
    from src.models.intervals import PredictionIntervals
    from src.models.backtesting import Backtester
    from src.visualization.dashboard import VisualizationEngine
    MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"Module import error: {e}. Running in demo mode.")
    MODULES_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="CortexX - Sales Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_data_cached(file) -> pd.DataFrame:
    """Cache data loading to improve performance."""
    try:
        if MODULES_AVAILABLE:
            collector = DataCollector()
            return collector.load_csv_data(file)
        else:
            return pd.read_csv(file)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def generate_sample_data_cached(periods: int = 365, products: int = 3) -> pd.DataFrame:
    """Cache sample data generation."""
    if MODULES_AVAILABLE:
        collector = DataCollector()
        return collector.generate_sample_data(periods, products)
    else:
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
        data = []
        for product_id in range(1, products + 1):
            base_sales = 1000 * product_id
            trend = np.linspace(0, 200, periods)
            seasonality = 300 * np.sin(2 * np.pi * np.arange(periods) / 365)
            noise = np.random.randn(periods) * 50
            sales = base_sales + trend + seasonality + noise

            for i, date in enumerate(dates):
                data.append({
                    'date': date,
                    'product_id': f'Product_{product_id}',
                    'sales': max(0, sales[i])
                })
        return pd.DataFrame(data)


class SalesForecastingApp:
    """Main application class for CortexX Sales Forecasting."""

    def __init__(self):
        """Initialize the application."""
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'current_data' not in st.session_state:
            st.session_state.current_data = None
        if 'date_column' not in st.session_state:
            st.session_state.date_column = None
        if 'value_column' not in st.session_state:
            st.session_state.value_column = None
        if 'trained_models' not in st.session_state:
            st.session_state.trained_models = {}
        if 'model_results' not in st.session_state:
            st.session_state.model_results = {}
        if 'best_model_name' not in st.session_state:
            st.session_state.best_model_name = None
        if 'optimization_results' not in st.session_state:
            st.session_state.optimization_results = {}
        if 'prediction_intervals' not in st.session_state:
            st.session_state.prediction_intervals = {}
        if 'backtest_results' not in st.session_state:
            st.session_state.backtest_results = {}
        if 'selected_features' not in st.session_state:
            st.session_state.selected_features = []
        if 'feature_importance' not in st.session_state:
            st.session_state.feature_importance = {}

    def run(self):
        """Run the main application."""
        # Header
        st.markdown('<div class="main-header">📊 CortexX Sales Forecasting Platform</div>', 
                   unsafe_allow_html=True)
        st.markdown("---")

        # Sidebar navigation
        with st.sidebar:
            st.image("https://via.placeholder.com/150x50/1f77b4/ffffff?text=CortexX", 
                    use_container_width=True)
            st.markdown("### Navigation")

            menu = st.radio(
                "Select Section",
                ["🏠 Home", "📊 Data Exploration", "🔧 Preprocessing", 
                 "⚙️ Feature Engineering", "🎯 Feature Selection",
                 "🤖 Model Training", "🔬 Hyperparameter Tuning",
                 "📉 Prediction Intervals", "🔄 Backtesting",
                 "📈 Results & Analysis"],
                label_visibility="collapsed"
            )

            st.markdown("---")
            st.markdown("### System Status")
            st.success("✅ All Modules Loaded" if MODULES_AVAILABLE else "⚠️ Demo Mode")

            if st.session_state.data_loaded:
                st.info(f"📁 Data: {len(st.session_state.current_data)} rows")

            if st.session_state.trained_models:
                st.success(f"🤖 Models: {len(st.session_state.trained_models)} trained")

            st.markdown("---")
            st.markdown("**Version:** 2.0.0")
            st.markdown("**M1-M4:** ✅ Complete")

        # Main content routing
        if menu == "🏠 Home":
            self.home_page()
        elif menu == "📊 Data Exploration":
            self.data_exploration_page()
        elif menu == "🔧 Preprocessing":
            self.preprocessing_page()
        elif menu == "⚙️ Feature Engineering":
            self.feature_engineering_page()
        elif menu == "🎯 Feature Selection":
            self.feature_selection_page()
        elif menu == "🤖 Model Training":
            self.model_training_page()
        elif menu == "🔬 Hyperparameter Tuning":
            self.hyperparameter_tuning_page()
        elif menu == "📉 Prediction Intervals":
            self.prediction_intervals_page()
        elif menu == "🔄 Backtesting":
            self.backtesting_page()
        elif menu == "📈 Results & Analysis":
            self.results_page()


    def home_page(self):
        """Home page with data upload."""
        st.header("🏠 Welcome to CortexX Sales Forecasting")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📤 Upload Your Data")
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="Upload your sales data in CSV format"
            )

            if uploaded_file is not None:
                try:
                    df = load_data_cached(uploaded_file)
                    if not df.empty:
                        st.session_state.current_data = df
                        st.session_state.data_loaded = True
                        self.detect_date_column(df)
                        st.success(f"✅ Data loaded: {len(df)} rows, {len(df.columns)} columns")
                        st.dataframe(df.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")

        with col2:
            st.subheader("🎲 Or Generate Sample Data")
            st.markdown("Test the platform with synthetic sales data")

            col_a, col_b = st.columns(2)
            with col_a:
                periods = st.number_input("Days", 100, 1000, 365)
            with col_b:
                products = st.number_input("Products", 1, 10, 3)

            if st.button("Generate Sample Data", type="primary"):
                with st.spinner("Generating data..."):
                    df = generate_sample_data_cached(periods, products)
                    st.session_state.current_data = df
                    st.session_state.data_loaded = True
                    self.detect_date_column(df)
                    st.success("✅ Sample data generated!")
                    st.dataframe(df.head(10), use_container_width=True)

        if not st.session_state.data_loaded:
            st.markdown("---")
            st.subheader("🚀 Platform Features")

            features = {
                "📊 Data Analysis": "Comprehensive EDA with statistical insights",
                "🔧 Preprocessing": "Missing values, outliers, scaling",
                "⚙️ Feature Engineering": "Time features, lags, rolling statistics",
                "🎯 Feature Selection": "4 selection methods (M2)",
                "🤖 ML Models": "9 algorithms including XGBoost, LightGBM",
                "🔬 Optimization": "Hyperparameter tuning with Optuna (M3)",
                "📉 Intervals": "95% prediction confidence bands (M3)",
                "🔄 Backtesting": "Walk-forward validation (M3)"
            }

            cols = st.columns(4)
            for idx, (title, desc) in enumerate(features.items()):
                with cols[idx % 4]:
                    st.markdown(f"**{title}**")
                    st.caption(desc)

    def detect_date_column(self, df: pd.DataFrame):
        """Auto-detect date column."""
        date_patterns = ['date', 'time', 'timestamp', 'datetime', 'day', 'month', 'year']

        for col in df.columns:
            if any(pattern in col.lower() for pattern in date_patterns):
                st.session_state.date_column = col
                try:
                    df[col] = pd.to_datetime(df[col])
                    st.info(f"📅 Detected date column: {col}")
                except:
                    pass
                return

        for col in df.columns:
            try:
                pd.to_datetime(df[col])
                st.session_state.date_column = col
                st.info(f"📅 Detected date column: {col}")
                return
            except:
                continue

    def data_exploration_page(self):
        """Data exploration and EDA."""
        st.header("📊 Data Exploration")

        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data first from the Home page")
            return

        df = st.session_state.current_data

        # Basic info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        with col4:
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            st.metric("Memory", f"{memory_mb:.2f} MB")

        # Data preview
        st.subheader("📋 Data Preview")
        view_option = st.radio("View", ["First 10", "Last 10", "Random 10"], horizontal=True)

        if view_option == "First 10":
            st.dataframe(df.head(10), use_container_width=True)
        elif view_option == "Last 10":
            st.dataframe(df.tail(10), use_container_width=True)
        else:
            st.dataframe(df.sample(min(10, len(df))), use_container_width=True)

        # Statistical summary
        if MODULES_AVAILABLE:
            st.subheader("📈 Statistical Analysis")
            if st.button("Generate Statistical Report"):
                with st.spinner("Analyzing..."):
                    explorer = DataExplorer()
                    stats = explorer.generate_summary_statistics(df)
                    st.json(stats)
        else:
            st.subheader("📈 Basic Statistics")
            st.dataframe(df.describe(), use_container_width=True)

    def preprocessing_page(self):
        """Data preprocessing interface."""
        st.header("🔧 Data Preprocessing")

        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data first")
            return

        df = st.session_state.current_data.copy()

        # Missing values
        st.subheader("🔍 Handle Missing Values")
        missing_cols = df.columns[df.isnull().any()].tolist()

        if missing_cols:
            st.warning(f"Found missing values in: {', '.join(missing_cols)}")
            strategy = st.selectbox("Strategy", ['interpolate', 'ffill', 'bfill', 'mean', 'median', 'drop'])

            if st.button("Apply Missing Value Treatment"):
                if MODULES_AVAILABLE:
                    preprocessor = DataPreprocessor()
                    df = preprocessor.handle_missing_values(df, strategy=strategy)
                else:
                    if strategy == 'drop':
                        df = df.dropna()
                    elif strategy in ['mean', 'median']:
                        df = df.fillna(df.mean() if strategy == 'mean' else df.median())

                st.session_state.current_data = df
                st.success("✅ Missing values handled!")
        else:
            st.success("✅ No missing values found")

        # Outlier removal
        st.subheader("🎯 Outlier Detection")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if numeric_cols:
            col = st.selectbox("Select column", numeric_cols)
            method = st.selectbox("Method", ['iqr', 'zscore', 'percentile'])

            if st.button("Remove Outliers"):
                if MODULES_AVAILABLE:
                    preprocessor = DataPreprocessor()
                    df_clean = preprocessor.remove_outliers(df, col, method=method)
                    removed = len(df) - len(df_clean)
                    st.info(f"Removed {removed} outlier rows")
                    st.session_state.current_data = df_clean
                else:
                    st.warning("Outlier removal requires full modules")

    def feature_engineering_page(self):
        """Feature engineering interface."""
        st.header("⚙️ Feature Engineering")

        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data first")
            return

        df = st.session_state.current_data.copy()

        if st.session_state.date_column is None:
            st.error("❌ No date column detected. Please specify manually.")
            date_col = st.selectbox("Select date column", df.columns)
            st.session_state.date_column = date_col
        else:
            date_col = st.session_state.date_column

        st.info(f"📅 Using date column: {date_col}")

        if not MODULES_AVAILABLE:
            st.warning("⚠️ Feature engineering requires full modules")
            return

        engineer = FeatureEngineer()

        # Time features
        st.subheader("📅 Time-Based Features")
        if st.button("Create Time Features"):
            with st.spinner("Creating time features..."):
                df = engineer.create_time_features(df, date_col)
                st.session_state.current_data = df
                st.success(f"✅ Added time features! Now {len(df.columns)} columns")
                st.dataframe(df.head(), use_container_width=True)

        # Lag features
        st.subheader("🔄 Lag Features")
        value_col = st.selectbox("Select value column", 
                                df.select_dtypes(include=[np.number]).columns)
        lags = st.multiselect("Lag periods", [1, 7, 14, 30, 90], default=[1, 7, 30])

        if st.button("Create Lag Features"):
            with st.spinner("Creating lag features..."):
                df = engineer.create_lag_features(df, value_col, lags=lags)
                st.session_state.current_data = df
                st.success(f"✅ Added lag features!")

        # Rolling features
        st.subheader("📊 Rolling Statistics")
        windows = st.multiselect("Window sizes", [7, 14, 30, 90], default=[7, 30])

        if st.button("Create Rolling Features"):
            with st.spinner("Creating rolling features..."):
                df = engineer.create_rolling_features(df, value_col, windows=windows)
                st.session_state.current_data = df
                st.success(f"✅ Added rolling features!")


    def feature_selection_page(self):
        """Feature Selection Interface - M2 (ALL ERRORS FIXED!)"""
        st.header("🎯 Feature Selection")
        st.markdown("**Milestone 2:** Select the most important features")

        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data first")
            return

        if not MODULES_AVAILABLE:
            st.warning("⚠️ Feature selection requires full modules")
            return

        df = st.session_state.current_data
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            st.error("❌ Need at least 2 numeric columns")
            return

        st.info("🎯 Select features using multiple methods")

        tab1, tab2, tab3 = st.tabs([
            "📊 Calculate Importance", "🎯 Select Features", "🔗 Remove Correlated"
        ])

        # Tab 1: Calculate Feature Importance
        with tab1:
            st.subheader("Calculate Feature Importance")
            target_col = st.selectbox("Target Column", numeric_cols, key="fi_target")
            # ✅ FIXED: Use correct method names from YOUR selection.py
            importance_method = st.selectbox("Method", ['randomforest', 'fregression', 'mutualinfo'])

            if st.button("Calculate Importance", type="primary"):
                with st.spinner("Calculating..."):
                    try:
                        selector = FeatureSelector()

                        # ✅ FIXED ERROR 1: Use calculate_feature_importance(df, target_col, method)
                        importance_df = selector.calculate_feature_importance(
                            df, target_col, importance_method
                        )

                        if not importance_df.empty:
                            st.session_state.feature_importance = importance_df.set_index('feature')['importance'].to_dict()
                            st.success(f"✅ Calculated importance for {len(importance_df)} features!")

                            # Show results
                            fig = px.bar(importance_df.head(15), x='importance', y='feature',
                                       orientation='h', title="Top 15 Features by Importance")
                            st.plotly_chart(fig, use_container_width=True)

                            st.dataframe(importance_df, use_container_width=True)
                        else:
                            st.warning("No importance scores calculated")

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

        # Tab 2: Select Top N Features
        with tab2:
            st.subheader("Select Top N Features")
            target_col = st.selectbox("Target Column", numeric_cols, key="sel_target")
            n_features = st.number_input("Number of features", 1, len(numeric_cols)-1, 
                                        min(10, len(numeric_cols)-1))
            # ✅ FIXED: Use correct method names
            select_method = st.selectbox("Selection Method", 
                                        ['randomforest', 'fregression', 'mutualinfo'], 
                                        key="sel_method")

            if st.button("Select Features", type="primary"):
                with st.spinner("Selecting..."):
                    try:
                        selector = FeatureSelector()

                        # ✅ FIXED ERROR 2, 3, 4: Use select_features(df, target_col, n_features, method)
                        selected_features = selector.select_features(
                            df, target_col, n_features, select_method
                        )

                        if selected_features:
                            st.session_state.selected_features = selected_features
                            st.success(f"✅ Selected {len(selected_features)} features!")
                            st.write("**Selected Features:**")
                            st.write(", ".join(selected_features))
                        else:
                            st.warning("No features selected")

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

        # Tab 3: Remove Correlated Features
        with tab3:
            st.subheader("Remove Highly Correlated Features")
            threshold = st.slider("Correlation Threshold", 0.5, 0.99, 0.95, 0.01,
                                help="Features with correlation above this will be removed")

            if st.button("Remove Correlated", type="primary"):
                with st.spinner("Analyzing..."):
                    try:
                        selector = FeatureSelector()

                        # ✅ FIXED: Use remove_correlated_features(df, threshold)
                        features_to_keep = selector.remove_correlated_features(df, threshold)

                        if features_to_keep:
                            removed = len(df.columns) - len(features_to_keep)
                            st.success(f"✅ Removed {removed} highly correlated features!")
                            st.info(f"📊 {len(features_to_keep)} features remaining")

                            st.session_state.selected_features = features_to_keep

                            with st.expander("Show Features to Keep"):
                                st.write(", ".join(features_to_keep))
                        else:
                            st.warning("No features to remove")

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

        # Summary
        if st.session_state.selected_features:
            st.markdown("---")
            st.subheader("✅ Current Selection")
            st.info(f"**{len(st.session_state.selected_features)}** features selected")

            if st.button("💾 Apply Selection to Dataset"):
                cols_to_keep = [c for c in st.session_state.selected_features if c in df.columns]
                if st.session_state.date_column and st.session_state.date_column in df.columns:
                    if st.session_state.date_column not in cols_to_keep:
                        cols_to_keep.append(st.session_state.date_column)

                df_selected = df[cols_to_keep]
                st.session_state.current_data = df_selected
                st.success("✅ Feature selection applied to dataset!")
                st.balloons()


    def model_training_page(self):
        """Model training interface."""
        st.header("🤖 Model Training")

        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data first")
            return

        if not MODULES_AVAILABLE:
            st.warning("⚠️ Model training requires full modules")
            return

        df = st.session_state.current_data

        # Configuration
        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox("Target Column", 
                                     df.select_dtypes(include=[np.number]).columns)
            st.session_state.value_column = target_col
        with col2:
            test_size = st.slider("Test Size (%)", 10, 40, 20) / 100

        # Model selection
        st.subheader("Select Models to Train")
        models_available = ['XGBoost', 'LightGBM', 'Random Forest', 'CatBoost',
                          'Lasso', 'Ridge', 'Decision Tree', 'KNN', 'SVR']

        selected_models = st.multiselect("Models", models_available, 
                                        default=['XGBoost', 'LightGBM', 'Random Forest'])

        if st.button("🚀 Train Models", type="primary"):
            if not selected_models:
                st.error("Please select at least one model")
                return

            trainer = ModelTrainer()
            date_col = st.session_state.date_column

            with st.spinner("Training models..."):
                progress_bar = st.progress(0)

                for idx, model_name in enumerate(selected_models):
                    st.info(f"Training {model_name}...")

                    try:
                        train_df, test_df = trainer.train_test_split(df, date_col, target_col, test_size)

                        model = None
                        results = None

                        # Train based on model type
                        if model_name == 'XGBoost':
                            model, results = trainer.train_xgboost(train_df, test_df, date_col, target_col)
                        elif model_name == 'LightGBM':
                            model, results = trainer.train_lightgbm(train_df, test_df, date_col, target_col)
                        elif model_name == 'Random Forest':
                            model, results = trainer.train_random_forest(train_df, test_df, date_col, target_col)
                        elif model_name == 'CatBoost':
                            model, results = trainer.train_catboost(train_df, test_df, date_col, target_col)
                        elif model_name == 'Lasso':
                            model, results = trainer.train_lasso(train_df, test_df, date_col, target_col)
                        elif model_name == 'Ridge':
                            model, results = trainer.train_ridge(train_df, test_df, date_col, target_col)
                        elif model_name == 'Decision Tree':
                            model, results = trainer.train_decision_tree(train_df, test_df, date_col, target_col)
                        elif model_name == 'KNN':
                            model, results = trainer.train_knn(train_df, test_df, date_col, target_col)
                        elif model_name == 'SVR':
                            model, results = trainer.train_svr(train_df, test_df, date_col, target_col)
                        else:
                            st.warning(f"⚠️ {model_name} not implemented")
                            continue

                        # Store results
                        if results is not None:
                            st.session_state.trained_models[model_name] = model
                            st.session_state.model_results[model_name] = results
                            st.success(f"✅ {model_name} trained!")
                        else:
                            st.warning(f"⚠️ {model_name} returned no results")

                    except Exception as e:
                        st.error(f"❌ {model_name} failed: {str(e)}")
                        logger.error(f"Model training error: {str(e)}")

                    progress_bar.progress((idx + 1) / len(selected_models))

                st.balloons()
                st.success("🎉 Training complete!")

    def hyperparameter_tuning_page(self):
        """Hyperparameter optimization - M3."""
        st.header("🔬 Hyperparameter Optimization")
        st.markdown("**Milestone 3:** Automatic tuning with Optuna")

        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data first")
            return

        if not MODULES_AVAILABLE:
            st.warning("⚠️ Optimization requires full modules")
            return

        df = st.session_state.current_data

        # Configuration
        col1, col2, col3 = st.columns(3)
        with col1:
            model_type = st.selectbox("Model", ['xgboost', 'lightgbm', 'random_forest'])
        with col2:
            n_trials = st.slider("Trials", 10, 100, 50)
        with col3:
            cv_splits = st.slider("CV Splits", 2, 5, 3)

        target_col = st.selectbox("Target Column", 
                                 df.select_dtypes(include=[np.number]).columns)

        if st.button("🎯 Start Optimization", type="primary"):
            with st.spinner(f"Optimizing {model_type} with {n_trials} trials..."):
                try:
                    X = df.select_dtypes(include=[np.number]).drop(columns=[target_col])
                    y = df[target_col]

                    optimizer = HyperparameterOptimizer(n_trials=n_trials, cv_splits=cv_splits)
                    result = optimizer.optimize_model(model_type, X, y, metric='rmse')

                    if 'error' not in result:
                        st.session_state.optimization_results[model_type] = result

                        st.success(f"✅ Optimization complete!")
                        st.metric("Best RMSE", f"{result['best_score']:.4f}")

                        st.subheader("🎯 Best Parameters")
                        st.json(result['best_params'])

                        # Plot history
                        if 'optimization_history' in result:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                y=result['optimization_history'],
                                mode='lines+markers',
                                name='RMSE'
                            ))
                            fig.update_layout(
                                title="Optimization History",
                                xaxis_title="Trial",
                                yaxis_title="RMSE",
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"❌ Optimization failed: {result.get('error')}")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    def prediction_intervals_page(self):
        """Prediction Intervals - M3 (NEW!)"""
        st.header("📉 Prediction Intervals")
        st.markdown("**Milestone 3:** Calculate 95% confidence bands")

        if not st.session_state.trained_models:
            st.warning("⚠️ Please train models first")
            return

        if not MODULES_AVAILABLE:
            st.warning("⚠️ Prediction intervals require full modules")
            return

        st.info("🎯 Prediction intervals provide uncertainty estimates")

        # Select model
        model_name = st.selectbox("Select Model", list(st.session_state.trained_models.keys()))

        # Configuration
        method = st.selectbox("Interval Method", ['residual', 'bootstrap', 'quantile'])
        confidence = st.slider("Confidence Level", 0.80, 0.99, 0.95, 0.01)

        if st.button("📊 Calculate Intervals", type="primary"):
            with st.spinner("Calculating intervals..."):
                try:
                    model = st.session_state.trained_models[model_name]
                    results = st.session_state.model_results[model_name]

                    y_train = results['y_train']
                    y_train_pred = results['train_predictions']
                    y_test_pred = results['test_predictions']

                    pi_calc = PredictionIntervals(confidence_level=confidence)

                    if method == 'residual':
                        intervals = pi_calc.calculate_residual_intervals(
                            y_train, y_train_pred, y_test_pred
                        )
                    elif method == 'quantile':
                        intervals = pi_calc.calculate_quantile_intervals(
                            y_train, y_train_pred, y_test_pred
                        )
                    else:
                        intervals = pi_calc.calculate_residual_intervals(
                            y_train, y_train_pred, y_test_pred
                        )

                    st.session_state.prediction_intervals[model_name] = intervals

                    st.success("✅ Intervals calculated!")

                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Confidence", f"{confidence*100:.0f}%")
                    with col2:
                        avg_width = np.mean(intervals['upper_bound'] - intervals['lower_bound'])
                        st.metric("Avg Width", f"{avg_width:.2f}")
                    with col3:
                        st.metric("Method", method.capitalize())

                    # Visualization
                    st.subheader("📈 Predictions with Confidence Bands")

                    y_test = results['y_test']
                    test_dates = results.get('test_dates', range(len(y_test)))

                    fig = go.Figure()

                    # Actual
                    fig.add_trace(go.Scatter(
                        x=test_dates,
                        y=y_test,
                        name='Actual',
                        line=dict(color='blue', width=2)
                    ))

                    # Predicted
                    fig.add_trace(go.Scatter(
                        x=test_dates,
                        y=intervals['predictions'],
                        name='Predicted',
                        line=dict(color='red', width=2)
                    ))

                    # Confidence bands
                    fig.add_trace(go.Scatter(
                        x=test_dates,
                        y=intervals['upper_bound'],
                        name=f'Upper {confidence*100:.0f}%',
                        line=dict(dash='dash', color='gray')
                    ))

                    fig.add_trace(go.Scatter(
                        x=test_dates,
                        y=intervals['lower_bound'],
                        name=f'Lower {confidence*100:.0f}%',
                        fill='tonexty',
                        line=dict(dash='dash', color='gray'),
                        fillcolor='rgba(128,128,128,0.2)'
                    ))

                    fig.update_layout(
                        title=f"Predictions with {confidence*100:.0f}% Confidence Intervals",
                        height=500
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Coverage
                    if len(y_test) == len(intervals['predictions']):
                        coverage = pi_calc.evaluate_interval_coverage(
                            y_test, intervals['lower_bound'], intervals['upper_bound']
                        )

                        st.subheader("📊 Coverage Analysis")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Actual Coverage", f"{coverage['coverage_percentage']:.1f}%")
                        with col_b:
                            st.metric("Expected Coverage", f"{coverage['expected_coverage']*100:.1f}%")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    def backtesting_page(self):
        """Backtesting - M3 (NEW!)"""
        st.header("🔄 Backtesting & Walk-Forward Validation")
        st.markdown("**Milestone 3:** Evaluate model with realistic time-series splitting")

        if not st.session_state.trained_models:
            st.warning("⚠️ Please train models first")
            return

        if not MODULES_AVAILABLE:
            st.warning("⚠️ Backtesting requires full modules")
            return

        st.info("🔄 Test how your model performs on sequential periods")

        # Configuration
        col1, col2, col3 = st.columns(3)
        with col1:
            initial_train_size = st.number_input("Initial Train Size", 50, 500, 100)
        with col2:
            test_size = st.number_input("Test Size", 10, 100, 30)
        with col3:
            step_size = st.number_input("Step Size", 10, 100, 30)

        window_type = st.radio("Window Type", ['expanding', 'rolling'])

        # Model selection
        models_to_test = st.multiselect(
            "Select models for backtesting",
            list(st.session_state.trained_models.keys()),
            default=[list(st.session_state.trained_models.keys())[0]]
        )

        if st.button("🔄 Run Backtesting", type="primary"):
            with st.spinner("Running walk-forward validation..."):
                try:
                    df = st.session_state.current_data
                    target_col = st.session_state.value_column
                    date_col = st.session_state.date_column

                    if target_col is None:
                        st.error("Please specify target column")
                        return

                    X = df.select_dtypes(include=[np.number]).drop(columns=[target_col])
                    y = df[target_col]
                    dates = df[date_col] if date_col else pd.Series(range(len(df)))

                    backtester = Backtester(
                        initial_train_size=initial_train_size,
                        test_size=test_size,
                        step_size=step_size,
                        window_type=window_type
                    )

                    backtest_results = {}
                    progress = st.progress(0)

                    for idx, model_name in enumerate(models_to_test):
                        st.info(f"Backtesting {model_name}...")

                        model = st.session_state.trained_models[model_name]
                        result = backtester.backtest_model(model, X, y, dates)

                        if 'error' not in result:
                            backtest_results[model_name] = result
                            st.success(f"✅ {model_name} backtested")
                        else:
                            st.warning(f"⚠️ {model_name}: {result['error']}")

                        progress.progress((idx + 1) / len(models_to_test))

                    st.session_state.backtest_results = backtest_results

                    if backtest_results:
                        st.success("🎉 Backtesting complete!")

                        # Comparison
                        st.subheader("🏆 Model Comparison")

                        comparison_data = []
                        for model_name, result in backtest_results.items():
                            metrics = result['aggregate_metrics']
                            comparison_data.append({
                                'Model': model_name,
                                'Overall RMSE': metrics['overall_rmse'],
                                'Avg RMSE': metrics['avg_rmse'],
                                'Overall R²': metrics['overall_r2'],
                                'Windows': result['n_windows']
                            })

                        comparison_df = pd.DataFrame(comparison_data).sort_values('Overall RMSE')
                        st.dataframe(comparison_df, use_container_width=True)

                        # Visualization
                        st.subheader("📈 Backtest Results")

                        for model_name, result in backtest_results.items():
                            with st.expander(f"📊 {model_name}"):
                                plot_df = backtester.create_forecast_plot_data(result)

                                if not plot_df.empty and 'date' in plot_df.columns:
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=plot_df['date'],
                                        y=plot_df['actual'],
                                        name='Actual',
                                        line=dict(color='blue')
                                    ))
                                    fig.add_trace(go.Scatter(
                                        x=plot_df['date'],
                                        y=plot_df['predicted'],
                                        name='Predicted',
                                        line=dict(color='red')
                                    ))
                                    fig.update_layout(title=f"{model_name} Results", height=400)
                                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")


    def results_page(self):
        """Results and visualization page - ALL VISUALIZATION ERRORS FIXED!"""
        st.header("📈 Results & Analysis")

        if not st.session_state.trained_models:
            st.warning("⚠️ No trained models yet. Please train models first.")
            return

        if not MODULES_AVAILABLE:
            st.warning("⚠️ Analysis requires full modules")
            return

        # Model comparison
        st.subheader("🏆 Model Performance Comparison")

        evaluator = ModelEvaluator()
        comparison = evaluator.compare_models(st.session_state.model_results)

        if not comparison.empty:
            st.dataframe(comparison, use_container_width=True)

            best_model = comparison.iloc[0]['model']
            st.session_state.best_model_name = best_model
            st.success(f"🥇 Best Model: **{best_model}**")

            # Visualizations
            st.subheader("📊 Visualizations")

            tab1, tab2, tab3 = st.tabs(["📈 Forecasts", "📊 Model Comparison", "📉 Residuals"])

            # Tab 1: Forecast Comparison
            with tab1:
                selected_model = st.selectbox("Select model", 
                                             list(st.session_state.model_results.keys()))

                if selected_model and selected_model in st.session_state.model_results:
                    results = st.session_state.model_results[selected_model]

                    try:
                        visualizer = VisualizationEngine()

                        # ✅ FIXED ERROR 5: Extract data from results and pass correct parameters
                        y_test = results.get('y_test', np.array([]))
                        y_test_pred = results.get('test_predictions', np.array([]))
                        test_dates = results.get('test_dates', np.arange(len(y_test)))

                        if len(y_test) > 0 and len(y_test_pred) > 0:
                            # ✅ CORRECT METHOD: create_forecast_comparison_plot(
                            #       actual_dates, actual_values, forecast_dates, forecast_values, model_name)
                            fig = visualizer.create_forecast_comparison_plot(
                                actual_dates=test_dates,
                                actual_values=y_test,
                                forecast_dates=test_dates,
                                forecast_values=y_test_pred,
                                model_name=selected_model
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # Show metrics
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("RMSE", f"{results.get('test_rmse', 0):.4f}")
                            col2.metric("MAE", f"{results.get('test_mae', 0):.4f}")
                            col3.metric("R²", f"{results.get('test_r2', 0):.4f}")
                            col4.metric("MAPE", f"{results.get('test_mape', 0):.2f}%")
                        else:
                            st.warning("⚠️ No test data available for visualization")

                    except Exception as e:
                        st.error(f"Visualization error: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

            # Tab 2: Model Comparison Chart
            with tab2:
                try:
                    visualizer = VisualizationEngine()

                    # ✅ FIXED: Add metric selection
                    metric = st.selectbox("Select Metric", ['rmse', 'mae', 'r2'], key="comp_metric")

                    if metric in comparison.columns:
                        # ✅ CORRECT METHOD: create_model_comparison_plot(comparison_df, metric)
                        fig = visualizer.create_model_comparison_plot(comparison, metric)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"Metric {metric} not found in comparison data")
                        st.info("Available metrics: " + ", ".join(comparison.columns.tolist()))

                except Exception as e:
                    st.error(f"Visualization error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

            # Tab 3: Residual Analysis
            with tab3:
                selected_residual = st.selectbox("Select Model for Residual Analysis", 
                                                list(st.session_state.model_results.keys()),
                                                key="residual_model")

                if selected_residual and selected_residual in st.session_state.model_results:
                    results = st.session_state.model_results[selected_residual]

                    try:
                        y_test = results.get('y_test', np.array([]))
                        y_test_pred = results.get('test_predictions', np.array([]))

                        if len(y_test) > 0 and len(y_test_pred) > 0:
                            visualizer = VisualizationEngine()

                            # ✅ CORRECT METHOD: create_residual_analysis_plot(actual_values, predicted_values)
                            fig = visualizer.create_residual_analysis_plot(y_test, y_test_pred)
                            st.plotly_chart(fig, use_container_width=True)

                            # Residual statistics
                            residuals = y_test - y_test_pred
                            col_a, col_b, col_c, col_d = st.columns(4)
                            col_a.metric("Mean Residual", f"{np.mean(residuals):.4f}")
                            col_b.metric("Std Residual", f"{np.std(residuals):.4f}")
                            col_c.metric("Min Residual", f"{np.min(residuals):.4f}")
                            col_d.metric("Max Residual", f"{np.max(residuals):.4f}")
                        else:
                            st.warning("⚠️ No test data available for residual analysis")

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

            # Export
            st.subheader("📥 Export Results")
            if st.button("📥 Download Comparison CSV"):
                csv = comparison.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="model_comparison.csv",
                    mime="text/csv"
                )


def main():
    """Main entry point."""
    try:
        app = SalesForecastingApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()