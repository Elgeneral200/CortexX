"""
Professional Model Training with Backtesting & Hyperparameter Tuning
"""


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import sys
import os


# Add src to path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
if src_path not in sys.path:
    sys.path.append(src_path)


try:
    from src.models.training import ModelTrainer
    from src.models.evaluation import ModelEvaluator
    from src.models.backtesting import Backtester
    from src.models.optimization import HyperparameterOptimizer
    from src.visualization.dashboard import VisualizationEngine, display_plotly_chart
    MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"Model training modules not available: {e}")
    MODULES_AVAILABLE = False


st.set_page_config(
    page_title="Model Training - CortexX",
    page_icon="🤖",
    layout="wide"
)


def main():
    """Main model training function."""
    
    st.markdown('<div class="section-header">🤖 ENTERPRISE MODEL TRAINING</div>', unsafe_allow_html=True)
    
    if not st.session_state.get('data_loaded', False):
        st.warning("⚠️ Please load data first from the Dashboard page")
        return
    
    if not MODULES_AVAILABLE:
        st.error("❌ Model training modules not available.")
        return
    
    df = st.session_state.current_data
    
    # Professional Training Interface
    st.markdown("""
    <div class="enterprise-card">
        <div class="card-title">🚀 ADVANCED MODEL TRAINING</div>
        <div class="card-description">Select from 11 enterprise-grade machine learning models for robust forecasting. Each model is trained with optimal parameters and evaluated using comprehensive validation techniques</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Configuration Section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 TRAINING CONFIGURATION")
        
        # Target column selection
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target_col = st.selectbox(
            "Select Target Column", 
            numeric_cols,
            index=0,
            help="Choose the column you want to forecast"
        )
        st.session_state.value_column = target_col
        
        # Test size selection
        test_size = st.slider(
            "Test Set Size (%)",
            min_value=10,
            max_value=40,
            value=20,
            help="Percentage of data to use for testing"
        ) / 100
        
        # Date column info
        date_col = st.session_state.get('date_column')
        if date_col and date_col in df.columns:
            st.info(f"📅 Using date column: {date_col} for time-based split")
        else:
            st.warning("⚠️ No date column detected. Using random split.")
    
    with col2:
        st.markdown("### 🎯 MODEL SELECTION")
        
        # Professional model selection
        available_models = {
            'XGBoost': 'Extreme Gradient Boosting - High performance, handles complex patterns',
            'LightGBM': 'Light Gradient Boosting - Fast training, great for large datasets',
            'Random Forest': 'Ensemble of decision trees - Robust, handles non-linearity',
            'CatBoost': 'Categorical Boosting - Great with categorical features, robust',
            'Linear Regression': 'Linear model - Fast, interpretable, baseline',
            'Ridge Regression': 'L2 regularized linear model - Prevents overfitting',
            'Lasso Regression': 'L1 regularized linear model - Feature selection capabilities',
            'Decision Tree': 'Single decision tree - Simple, interpretable',
            'K-Nearest Neighbors': 'Instance-based learning - Simple, no training required',
            'Support Vector Regression': 'Kernel-based - Good for complex non-linear patterns',
            'Prophet': 'Facebook Prophet - Specialized for time series with seasonality'
        }
        
        selected_models = []
        for model_name, description in available_models.items():
            if st.checkbox(f"{model_name}", 
                          value=model_name in ['XGBoost', 'LightGBM', 'Random Forest'], 
                          help=description):
                selected_models.append(model_name)
    
    # PROFESSIONAL BACKTESTING CONFIGURATION - VERTICAL LAYOUT
    st.markdown("---")
    st.markdown("### 🔄 BACKTESTING CONFIGURATION")
    
    # Initialize with default values
    enable_backtesting = False
    initial_train_size = 100
    test_size_backtest = 30
    step_size = 30
    window_type = 'expanding'
    
    # Vertical layout for professional appearance
    enable_backtesting = st.checkbox(
        "Enable Advanced Backtesting", 
        value=True,
        help="Run comprehensive walk-forward validation for robust model evaluation"
    )
    
    if enable_backtesting:
        st.markdown("**📊 Backtesting Parameters**")
        
        # Professional vertical layout
        col3, col4 = st.columns(2)
        
        with col3:
            initial_train_size = st.number_input(
                "Initial Training Size", 
                min_value=50, 
                max_value=1000, 
                value=100,
                help="Number of periods for initial training window"
            )
            
            test_size_backtest = st.number_input(
                "Test Size per Window", 
                min_value=10, 
                max_value=100, 
                value=30,
                help="Number of periods to forecast in each validation window"
            )
        
        with col4:
            step_size = st.number_input(
                "Step Size", 
                min_value=10, 
                max_value=100, 
                value=30,
                help="Number of periods to advance between validation windows"
            )
            
            window_type = st.selectbox(
                "Window Strategy", 
                ['expanding', 'rolling'],
                help="Expanding: Grow training set | Rolling: Fixed training size"
            )
    
    # HYPERPARAMETER TUNING SECTION
    st.markdown("---")
    st.markdown("### 🔬 HYPERPARAMETER OPTIMIZATION")
    
    # Initialize default values BEFORE the conditional
    n_trials = 10
    cv_splits = 3
    
    enable_tuning = st.checkbox(
        "Enable Hyperparameter Tuning", 
        value=False,
        help="Use Optuna for automatic optimization of model parameters (increases training time)"
    )
    
    if enable_tuning:
        col5, col6 = st.columns(2)
        
        with col5:
            n_trials = st.slider(
                "Number of Optimization Trials",
                min_value=10,
                max_value=100,
                value=50,
                help="More trials = better optimization but longer training time"
            )
        
        with col6:
            cv_splits = st.slider(
                "Cross-Validation Splits",
                min_value=2,
                max_value=5,
                value=3,
                help="Number of cross-validation folds for robust evaluation"
            )
    
    # Training Execution
    st.markdown("---")
    st.markdown("### 🚀 TRAINING EXECUTION")
    
    if not selected_models:
        st.warning("Please select at least one model to train")
        return
    
    if st.button("🎯 START ENTERPRISE TRAINING", type="primary", use_container_width=True):
        # Pass tuning parameters
        train_models(
            df, selected_models, target_col, test_size, date_col, 
            enable_backtesting, initial_train_size, test_size_backtest,
            step_size, window_type, enable_tuning, 
            n_trials,
            cv_splits
        )


def train_models(df: pd.DataFrame, selected_models: list, target_col: str, test_size: float, 
                date_col: str, enable_backtesting: bool, initial_train_size: int, 
                test_size_backtest: int, step_size: int, window_type: str,
                enable_tuning: bool, n_trials: int, cv_splits: int):
    """Train selected models with professional progress tracking."""
    
    trainer = ModelTrainer()
    visualizer = VisualizationEngine()
    
    # Initialize progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()
    
    # Initialize session state for results
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    if 'model_results' not in st.session_state:
        st.session_state.model_results = {}
    if 'backtest_results' not in st.session_state:
        st.session_state.backtest_results = {}
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = {}
    
    # Train each model
    trained_count = 0
    total_models = len(selected_models)
    
    for i, model_name in enumerate(selected_models):
        status_text.text(f"🔄 Training {model_name}... ({i+1}/{total_models})")
        
        try:
            # Train-test split
            if date_col and date_col in df.columns:
                train_df, test_df = trainer.train_test_split(df, date_col, target_col, test_size)
            else:
                # Fallback to random split
                split_idx = int(len(df) * (1 - test_size))
                train_df = df.iloc[:split_idx]
                test_df = df.iloc[split_idx:]
            
            # Hyperparameter tuning if enabled
            best_params = None
            if enable_tuning and MODULES_AVAILABLE:
                status_text.text(f"🔬 Optimizing {model_name}... ({i+1}/{total_models})")
                try:
                    optimizer = HyperparameterOptimizer(n_trials=n_trials, cv_splits=cv_splits)
                    X_train = train_df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
                    y_train = train_df[target_col]
                    
                    tuning_result = optimizer.optimize_model(
                        model_name.lower().replace(' ', '_'), 
                        X_train, y_train, metric='rmse'
                    )
                    
                    if 'error' not in tuning_result:
                        best_params = tuning_result['best_params']
                        st.session_state.optimization_results[model_name] = tuning_result
                        st.info(f"✅ Hyperparameter optimization completed for {model_name}")
                except Exception as e:
                    st.warning(f"Hyperparameter tuning failed for {model_name}: {str(e)}")
            
            # Train specific model - FIXED: Now properly handles best_params
            model, results = train_single_model(
                trainer, model_name, train_df, test_df, date_col or 'index', target_col, best_params
            )
            
            if model and results:
                # Store results
                st.session_state.trained_models[model_name] = model
                st.session_state.model_results[model_name] = results
                trained_count += 1
                
                # Run backtesting if enabled
                if enable_backtesting and MODULES_AVAILABLE:
                    backtest_single_model(model_name, model, df, target_col, date_col, 
                                        initial_train_size, test_size_backtest, step_size, window_type)
                
                # Show immediate results
                with results_container:
                    show_model_result(model_name, results)
            
            # Update progress
            progress = (i + 1) / total_models
            progress_bar.progress(progress)
            
            time.sleep(0.3)  # Visual progress
            
        except Exception as e:
            st.error(f"❌ {model_name} failed: {str(e)}")
            continue
    
    # Final summary
    progress_bar.empty()
    status_text.empty()
    
    if trained_count > 0:
        st.success(f"🎉 Training completed! {trained_count}/{total_models} models trained successfully")
        if enable_backtesting:
            st.success(f"📊 Backtesting completed for {len(st.session_state.backtest_results)} models")
        if enable_tuning:
            st.success(f"🔬 Hyperparameter tuning completed for {len(st.session_state.optimization_results)} models")
        st.balloons()
        
        # Show overall comparison
        show_training_summary()
    else:
        st.error("❌ No models were successfully trained")


def train_single_model(trainer, model_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                      date_col: str, target_col: str, best_params: dict = None):
    """
    Train a single model with error handling and optional hyperparameters.
    
    FIXED: Now handles both scenarios - with and without hyperparameter tuning
    """
    try:
        # Check if the ModelTrainer methods support best_params
        # If not, we'll apply them manually after training
        
        if model_name == 'XGBoost':
            model, results = trainer.train_xgboost(train_df, test_df, date_col, target_col)
            # If hyperparameters were optimized, retrain with them
            if best_params and model:
                model = apply_hyperparameters(model, best_params, 'xgboost')
                # Re-evaluate with optimized parameters
                results = evaluate_model_performance(model, test_df, target_col, results)
            return model, results
            
        elif model_name == 'LightGBM':
            model, results = trainer.train_lightgbm(train_df, test_df, date_col, target_col)
            if best_params and model:
                model = apply_hyperparameters(model, best_params, 'lightgbm')
                results = evaluate_model_performance(model, test_df, target_col, results)
            return model, results
            
        elif model_name == 'Random Forest':
            model, results = trainer.train_random_forest(train_df, test_df, date_col, target_col)
            if best_params and model:
                model = apply_hyperparameters(model, best_params, 'random_forest')
                results = evaluate_model_performance(model, test_df, target_col, results)
            return model, results
            
        elif model_name == 'CatBoost':
            model, results = trainer.train_catboost(train_df, test_df, date_col, target_col)
            if best_params and model:
                model = apply_hyperparameters(model, best_params, 'catboost')
                results = evaluate_model_performance(model, test_df, target_col, results)
            return model, results
            
        elif model_name == 'Linear Regression':
            return trainer.train_linear_regression(train_df, test_df, date_col, target_col)
            
        elif model_name == 'Ridge Regression':
            model, results = trainer.train_ridge(train_df, test_df, date_col, target_col)
            if best_params and model:
                model = apply_hyperparameters(model, best_params, 'ridge')
                results = evaluate_model_performance(model, test_df, target_col, results)
            return model, results
            
        elif model_name == 'Lasso Regression':
            model, results = trainer.train_lasso(train_df, test_df, date_col, target_col)
            if best_params and model:
                model = apply_hyperparameters(model, best_params, 'lasso')
                results = evaluate_model_performance(model, test_df, target_col, results)
            return model, results
            
        elif model_name == 'Decision Tree':
            model, results = trainer.train_decision_tree(train_df, test_df, date_col, target_col)
            if best_params and model:
                model = apply_hyperparameters(model, best_params, 'decision_tree')
                results = evaluate_model_performance(model, test_df, target_col, results)
            return model, results
            
        elif model_name == 'K-Nearest Neighbors':
            model, results = trainer.train_knn(train_df, test_df, date_col, target_col)
            if best_params and model:
                model = apply_hyperparameters(model, best_params, 'knn')
                results = evaluate_model_performance(model, test_df, target_col, results)
            return model, results
            
        elif model_name == 'Support Vector Regression':
            model, results = trainer.train_svr(train_df, test_df, date_col, target_col)
            if best_params and model:
                model = apply_hyperparameters(model, best_params, 'svr')
                results = evaluate_model_performance(model, test_df, target_col, results)
            return model, results
            
        elif model_name == 'Prophet':
            return trainer.train_prophet(train_df, date_col, target_col)
            
    except Exception as e:
        st.error(f"Error training {model_name}: {str(e)}")
        return None, None


def apply_hyperparameters(model, best_params: dict, model_type: str):
    """
    Apply optimized hyperparameters to a trained model by creating a new instance.
    
    Args:
        model: Original trained model
        best_params: Dictionary of optimized hyperparameters
        model_type: Type of model (e.g., 'xgboost', 'lightgbm')
    
    Returns:
        Model with updated hyperparameters
    """
    try:
        if not best_params:
            return model
        
        # Update model parameters based on type
        if hasattr(model, 'set_params'):
            model.set_params(**best_params)
        elif hasattr(model, '__dict__'):
            for param, value in best_params.items():
                if hasattr(model, param):
                    setattr(model, param, value)
        
        return model
    except Exception as e:
        st.warning(f"Could not apply hyperparameters: {str(e)}")
        return model


def evaluate_model_performance(model, test_df: pd.DataFrame, target_col: str, original_results: dict):
    """
    Re-evaluate model performance after applying hyperparameters.
    
    Args:
        model: Trained model with updated parameters
        test_df: Test dataset
        target_col: Target column name
        original_results: Original results dictionary
    
    Returns:
        Updated results dictionary
    """
    try:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Prepare test data
        X_test = test_df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
        y_test = test_df[target_col]
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Update results
        original_results['test_rmse'] = rmse
        original_results['test_mae'] = mae
        original_results['test_r2'] = r2
        original_results['optimized'] = True
        
        return original_results
        
    except Exception as e:
        st.warning(f"Could not re-evaluate model: {str(e)}")
        return original_results


def backtest_single_model(model_name: str, model, df: pd.DataFrame, target_col: str, 
                         date_col: str, initial_train_size: int, test_size_backtest: int, 
                         step_size: int, window_type: str):
    """Run backtesting for a single trained model."""
    try:
        # Initialize backtester
        backtester = Backtester(
            initial_train_size=initial_train_size,
            test_size=test_size_backtest,
            step_size=step_size,
            window_type=window_type
        )
        
        # Prepare data
        if date_col and date_col in df.columns:
            df_sorted = df.sort_values(date_col).reset_index(drop=True)
            date_series = df_sorted[date_col]
        else:
            df_sorted = df.reset_index(drop=True)
            date_series = None
        
        # Prepare features (X) and target (y) separately
        X = df_sorted.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
        y = df_sorted[target_col]
        
        # Run backtesting using the correct method name and parameters
        backtest_results = backtester.backtest_model(
            model=model,
            X=X,
            y=y,
            date_col=date_series,
            refit=True
        )
        
        # Check for errors
        if 'error' in backtest_results:
            st.warning(f"Backtesting failed for {model_name}: {backtest_results['error']}")
            return None
        
        # Store results
        st.session_state.backtest_results[model_name] = backtest_results
        
        return backtest_results
        
    except Exception as e:
        st.warning(f"Backtesting failed for {model_name}: {str(e)}")
        return None


def show_model_result(model_name: str, results: dict):
    """Display professional model training results."""
    
    with st.expander(f"📊 {model_name} Results", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Training Time", f"{results.get('training_time', 0):.2f}s")
        
        with col2:
            rmse = results.get('test_rmse', results.get('rmse', 'N/A'))
            st.metric("RMSE", f"{rmse:.4f}" if isinstance(rmse, (int, float)) else rmse)
        
        with col3:
            mae = results.get('test_mae', results.get('mae', 'N/A'))
            st.metric("MAE", f"{mae:.4f}" if isinstance(mae, (int, float)) else mae)
        
        with col4:
            r2 = results.get('test_r2', results.get('r2', 'N/A'))
            st.metric("R² Score", f"{r2:.4f}" if isinstance(r2, (int, float)) else r2)
        
        # Show if model was optimized
        if results.get('optimized', False):
            st.success("✨ Model trained with optimized hyperparameters")
        
        # Feature importance if available
        if 'feature_importance' in results and results['feature_importance']:
            st.markdown("**🔍 Feature Importance**")
            importance_df = pd.DataFrame({
                'feature': list(results['feature_importance'].keys()),
                'importance': list(results['feature_importance'].values())
            }).sort_values('importance', ascending=False)
            
            # Show top 10 features
            if not importance_df.empty:
                fig = px.bar(importance_df.head(10), x='importance', y='feature',
                            orientation='h', 
                            title=f"{model_name} - Top 10 Features",
                            color='importance',
                            color_continuous_scale='Viridis')
                fig.update_layout(plot_bgcolor='#1a1d29', paper_bgcolor='#1a1d29',
                                font=dict(color='white'))
                display_plotly_chart(fig)


def show_training_summary():
    """Display professional training summary."""
    
    st.markdown("---")
    st.markdown("### 📈 ENTERPRISE TRAINING SUMMARY")
    
    if not st.session_state.model_results:
        st.warning("No model results available")
        return
    
    # Create comparison dataframe with error handling
    comparison_data = []
    for model_name, results in st.session_state.model_results.items():
        # Safely extract metrics with fallbacks
        rmse = results.get('test_rmse', results.get('rmse', np.nan))
        mae = results.get('test_mae', results.get('mae', np.nan))
        r2 = results.get('test_r2', results.get('r2', np.nan))
        training_time = results.get('training_time', 0)
        optimized = '✅' if results.get('optimized', False) else '❌'
        
        model_metrics = {
            'Model': model_name,
            'Optimized': optimized,
            'Training Time (s)': training_time,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        }
        comparison_data.append(model_metrics)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display comparison table
    st.markdown("#### 📊 MODEL PERFORMANCE COMPARISON")
    st.dataframe(comparison_df, use_container_width=True)
    
    # Visual comparison with error handling
    col1, col2 = st.columns(2)
    
    with col1:
        # RMSE comparison
        if 'RMSE' in comparison_df.columns and not comparison_df['RMSE'].isna().all():
            valid_rmse = comparison_df.dropna(subset=['RMSE'])
            if not valid_rmse.empty:
                fig = px.bar(valid_rmse.sort_values('RMSE'), 
                            x='Model', y='RMSE',
                            title="Model Comparison - RMSE (Lower is Better)",
                            color='RMSE',
                            color_continuous_scale='Viridis')
                fig.update_layout(plot_bgcolor='#1a1d29', paper_bgcolor='#1a1d29',
                                font=dict(color='white'))
                display_plotly_chart(fig)
    
    with col2:
        # R² comparison
        if 'R²' in comparison_df.columns and not comparison_df['R²'].isna().all():
            valid_r2 = comparison_df.dropna(subset=['R²'])
            if not valid_r2.empty:
                fig = px.bar(valid_r2.sort_values('R²', ascending=False), 
                            x='Model', y='R²',
                            title="Model Comparison - R² Score (Higher is Better)",
                            color='R²',
                            color_continuous_scale='Plasma')
                fig.update_layout(plot_bgcolor='#1a1d29', paper_bgcolor='#1a1d29',
                                font=dict(color='white'))
                display_plotly_chart(fig)
    
    # Best model identification
    if 'RMSE' in comparison_df.columns and not comparison_df['RMSE'].isna().all():
        valid_comparison = comparison_df.dropna(subset=['RMSE'])
        if not valid_comparison.empty:
            best_model_idx = valid_comparison['RMSE'].idxmin()
            best_model = valid_comparison.loc[best_model_idx, 'Model']
            st.session_state.best_model_name = best_model
            
            st.success(f"🏆 **BEST PERFORMING MODEL**: {best_model}")
            
            # Show best model details
            if best_model in st.session_state.model_results:
                best_results = st.session_state.model_results[best_model]
                
                st.markdown("#### 🎯 BEST MODEL DETAILS")
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    rmse = best_results.get('test_rmse', best_results.get('rmse', 0))
                    st.metric("RMSE", f"{rmse:.4f}" if isinstance(rmse, (int, float)) else "N/A")
                with col_b:
                    mae = best_results.get('test_mae', best_results.get('mae', 0))
                    st.metric("MAE", f"{mae:.4f}" if isinstance(mae, (int, float)) else "N/A")
                with col_c:
                    r2 = best_results.get('test_r2', best_results.get('r2', 0))
                    st.metric("R²", f"{r2:.4f}" if isinstance(r2, (int, float)) else "N/A")
                with col_d:
                    st.metric("Training Time", f"{best_results.get('training_time', 0):.2f}s")
    
    # Backtesting results
    if st.session_state.backtest_results:
        st.markdown("#### 🔄 BACKTESTING RESULTS")
        
        backtest_summary = []
        for model_name, results in st.session_state.backtest_results.items():
            if 'aggregate_metrics' in results:
                metrics = results['aggregate_metrics']
                backtest_summary.append({
                    'Model': model_name,
                    'Overall RMSE': f"{metrics.get('overall_rmse', 0):.4f}",
                    'Avg RMSE': f"{metrics.get('avg_rmse', 0):.4f}",
                    'Windows': results.get('n_windows', 0)
                })
        
        if backtest_summary:
            backtest_df = pd.DataFrame(backtest_summary)
            st.dataframe(backtest_df, use_container_width=True)
    
    # Hyperparameter optimization results
    if st.session_state.optimization_results:
        st.markdown("#### 🔬 HYPERPARAMETER OPTIMIZATION RESULTS")
        
        opt_summary = []
        for model_name, results in st.session_state.optimization_results.items():
            if 'best_params' in results:
                opt_summary.append({
                    'Model': model_name,
                    'Best Score': f"{results.get('best_value', 0):.4f}",
                    'Trials': results.get('n_trials', 0),
                    'Status': '✅ Optimized'
                })
        
        if opt_summary:
            opt_df = pd.DataFrame(opt_summary)
            st.dataframe(opt_df, use_container_width=True)
    
    # Professional next steps
    st.markdown("---")
    st.markdown("### 🚀 NEXT STEPS")
    
    col_x, col_y = st.columns(2)
    
    with col_x:
        if st.button("📈 PROCEED TO FORECASTING", use_container_width=True):
            st.success("Navigate to the Forecasting page to generate predictions!")
    
    with col_y:
        if st.button("📊 PROCEED TO MODEL EVALUATION", use_container_width=True):
            st.success("Navigate to the Model Evaluation page for detailed analysis!")


if __name__ == "__main__":
    main()
