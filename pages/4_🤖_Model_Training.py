"""
CortexX - Enterprise Model Training Page

ENHANCED: Phase 2 Integration
- Uses cached singletons for performance
- StateManager for centralized state
- Config integration for settings
- Model Persistence (Save/Load/Manage)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import time
import sys
import os
import json

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# ✅ PHASE 1 ENHANCED IMPORTS
try:
    # Core utilities
    from src.utils.config import get_config
    from src.utils.state_manager import StateManager, is_data_loaded, get_current_data
    
    # Cached singletons (Phase 1 enhancement)
    from src.models.training import get_model_trainer
    from src.visualization.dashboard import get_visualizer, display_plotly_chart
    
    # Optional advanced modules
    from src.models.evaluation import ModelEvaluator
    from src.models.backtesting import Backtester
    from src.models.optimization import HyperparameterOptimizer
    
    MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Model training modules not available: {e}")
    MODULES_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="Model Training - CortexX",
    page_icon="🤖",
    layout="wide"
)
def render_no_features_warning():
    """Show warning when features aren't engineered"""
    st.error("❌ **No Engineered Features Found!**")
    st.markdown("""
    ### ⚠️ Critical: Engineer Features First
    
    **Current Status:**
    - Raw data loaded: ✅
    - Engineered features: ❌ **MISSING**
    
    **Why This Matters:**
    - Raw data: 10-15 columns → 60-70% accuracy ❌
    - Engineered data: 50-150+ features → 85-95% accuracy ✅
    
    **Impact on Your Models:**
    | Aspect | Without Features | With Features |
    |--------|------------------|---------------|
    | Columns | 10-15 | 150+ |
    | Accuracy | 60-70% | 85-95% |
    | Patterns | Basic | Complex + Seasonal |
    | Business Value | Low | High |
    
    **Action Required:**
    1. Navigate to **📊 Feature Engineering** page
    2. Click **"🚀 RUN FULL PIPELINE"**
    3. Wait ~30-60 seconds for features to be created
    4. Return here to train high-performance models
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➡️ Go to Feature Engineering", type="primary", use_container_width=True):
            st.switch_page("pages/3_-_Feature_Engineering.py")
    
    with col2:
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()


def main():
    """Main model training function with enterprise features."""
    
    st.markdown('<div class="section-header">🤖 ENTERPRISE MODEL TRAINING</div>', unsafe_allow_html=True)
    
    StateManager.initialize()
    
    # ✅ FIX 1: Check if data is loaded
    if not is_data_loaded():
        st.warning("⚠️ Please load data first from the Dashboard page")
        return
    
    if not MODULES_AVAILABLE:
        st.error("❌ Model training modules not available.")
        return
    
    # ✅ FIX 2: Check if features are engineered (CRITICAL!)
    if not StateManager.is_data_engineered():
        render_no_features_warning()
        return
    
    # ✅ FIX 3: Load ENGINEERED data (not raw!)
    df = StateManager.get_engineered_data()
    selected_features = StateManager.get('selected_features', [])
    
    # Show status
    st.success(f"✅ Using engineered data: {len(df):,} rows × {len(df.columns)} features")
    if selected_features:
        st.info(f"🎯 {len(selected_features)} features selected for optimal performance")
    else:
        st.info("🎯 No features selected for optimal performance")
    
    # Professional Training Interface
    st.markdown("""
    <div class="enterprise-card">
        <div class="card-title">🚀 ADVANCED MODEL TRAINING</div>
        <div class="card-description">Select from 9 enterprise-grade machine learning models for robust forecasting. Each model is trained with optimal parameters and evaluated using comprehensive validation techniques</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Configuration Section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 TRAINING CONFIGURATION")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target_col = st.selectbox(
            "Select Target Column", 
            numeric_cols,
            index=0,
            help="Choose the column you want to forecast"
        )
        StateManager.set('value_column', target_col)
        
        test_size = st.slider(
            "Test Set Size (%)",
            min_value=10,
            max_value=40,
            value=20,
            help="Percentage of data to use for testing"
        ) / 100
        
        date_col = StateManager.get('date_column')
        if date_col and date_col in df.columns:
            st.info(f"📅 Using date column: {date_col} for time-based split")
        else:
            st.warning("⚠️ No date column detected. Using random split.")
    
    with col2:
        st.markdown("### 🎯 MODEL SELECTION")
        
        available_models = {
            '🚀 XGBoost': 'Extreme Gradient Boosting - High performance, handles complex patterns excellently',
            '⚡ LightGBM': 'Light Gradient Boosting - Fast training, memory efficient, great for large datasets',
            '🌲 Random Forest': 'Ensemble of decision trees - Robust, handles non-linearity well',
            '🐱 CatBoost': 'Categorical Boosting - Excellent with categorical features, robust to overfitting',
            '📈 Linear Regression': 'Linear model - Fast, interpretable, good baseline',
            '🔷 Ridge Regression': 'L2 regularized linear model - Prevents overfitting, stable predictions',
            '🔶 Lasso Regression': 'L1 regularized linear model - Feature selection, sparse solutions',
            '🌳 Decision Tree': 'Single decision tree - Simple, interpretable, non-linear patterns',
            '📍 K-Nearest Neighbors': 'Instance-based learning - Simple, no training phase, local patterns'
        }
        
        selected_models = st.multiselect(
            "Choose models (select multiple for comparison)",
            options=list(available_models.keys()),
            default=['🚀 XGBoost', '⚡ LightGBM', '🌲 Random Forest'],
            format_func=lambda x: f"{x} - {available_models[x]}"
        )
        
        selected_model_names = [model.split(' ', 1)[1] for model in selected_models]
    
    # BACKTESTING CONFIGURATION
    st.markdown("---")
    st.markdown("### 🔄 BACKTESTING CONFIGURATION")
    
    enable_backtesting = st.checkbox(
        "Enable Advanced Backtesting", 
        value=False,
        help="Run comprehensive walk-forward validation for robust model evaluation"
    )
    
    if enable_backtesting:
        st.markdown("**📊 Backtesting Parameters**")
        
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
    else:
        initial_train_size = 100
        test_size_backtest = 30
        step_size = 30
        window_type = 'expanding'
    
    # HYPERPARAMETER TUNING
    st.markdown("---")
    st.markdown("### 🔬 HYPERPARAMETER OPTIMIZATION")
    
    enable_tuning = st.checkbox(
        "Enable Hyperparameter Tuning", 
        value=False,
        help="Use Optuna for automatic optimization of model parameters (increases training time)"
    )
    
    if enable_tuning:
        st.info("ℹ️ Note: Linear Regression models will use default parameters (optimization not applicable)")
        
        col5, col6 = st.columns(2)
        
        with col5:
            n_trials = st.slider(
                "Number of Optimization Trials",
                min_value=10,
                max_value=100,
                value=30,
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
    else:
        n_trials = 10
        cv_splits = 3
    
    # TRAINING EXECUTION
    st.markdown("---")
    st.markdown("### 🚀 TRAINING EXECUTION")
    
    if not selected_model_names:
        st.warning("⚠️ Please select at least one model to train")
        return
    
    st.markdown(f"""
    **Training Summary:**
    - 🎯 Models: {len(selected_model_names)} selected
    - 📊 Dataset: {len(df):,} rows
    - ✂️ Split: {int((1-test_size)*100)}% train / {int(test_size*100)}% test
    - {'🔬 Hyperparameter tuning: ENABLED' if enable_tuning else '⚡ Using default parameters'}
    - {'🔄 Backtesting: ENABLED' if enable_backtesting else '📈 Standard validation only'}
    """)
    
    if st.button("🎯 START ENTERPRISE TRAINING", type="primary", use_container_width=True):
        train_models(
            df=df,
            selected_models=selected_model_names,
            target_col=target_col,
            test_size=test_size,
            date_col=date_col,
            enable_backtesting=enable_backtesting,
            initial_train_size=initial_train_size,
            test_size_backtest=test_size_backtest,
            step_size=step_size,
            window_type=window_type,
            enable_tuning=enable_tuning,
            n_trials=n_trials,
            cv_splits=cv_splits
        )
    
    # ✅ TASK 3: MODEL MANAGEMENT SECTION
    st.markdown("---")
    config = get_config()
    model_registry_path = Path(config.model.model_registry_path)
    
    # Show if models have been trained or saved models exist
    if StateManager.get('trained_models') or (model_registry_path.exists() and list(model_registry_path.glob("*.pkl"))):
        show_model_management()


def train_models(df: pd.DataFrame, selected_models: list, target_col: str, test_size: float, 
                date_col: str, enable_backtesting: bool, initial_train_size: int, 
                test_size_backtest: int, step_size: int, window_type: str,
                enable_tuning: bool, n_trials: int, cv_splits: int):
    """Train selected models with professional progress tracking."""
    
    trainer = get_model_trainer()
    visualizer = get_visualizer()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()
    
    trained_models = StateManager.get('trained_models', {})
    model_results = StateManager.get('model_results', {})
    backtest_results = StateManager.get('backtest_results', {})
    optimization_results = StateManager.get('optimization_results', {})
    
    # ========================================
    # ✅ FEATURE SELECTION INTEGRATION
    # ========================================
    selected_features_list = StateManager.get('selected_features', [])
    
    # Prepare data with selected features (if available)
    if selected_features_list and len(selected_features_list) > 0:
        # Ensure target and date columns are included
        required_cols = [target_col]
        if date_col and date_col in df.columns:
            required_cols.append(date_col)
        
        # Combine selected features with required columns (remove duplicates)
        all_required_cols = list(set(selected_features_list + required_cols))
        available_cols = [col for col in all_required_cols if col in df.columns]
        
        df_for_training = df[available_cols].copy()
        
        st.success(f"✨ Training with {len(selected_features_list)} selected features (optimized from {len(df.columns)} total)")
        st.info("📊 Features selected for maximum predictive power across store-product combinations")
    else:
        df_for_training = df.copy()
        st.info(f"📊 Training with all {len(df.columns)} available features")
    
    trained_count = 0
    total_models = len(selected_models)
    
    for i, model_name in enumerate(selected_models):
        status_text.text(f"🔄 Training {model_name}... ({i+1}/{total_models})")
        
        try:
            # ✅ UPDATED: Train-test split using feature-selected data
            if date_col and date_col in df_for_training.columns:
                train_df, test_df = trainer.train_test_split(df_for_training, date_col, target_col, test_size)
            else:
                split_idx = int(len(df_for_training) * (1 - test_size))
                train_df = df_for_training.iloc[:split_idx]
                test_df = df_for_training.iloc[split_idx:]
            
            # ✅ NEW: Show training dimensions
            st.caption(f"📊 {model_name}: {len(train_df):,} train / {len(test_df):,} test samples, {len(train_df.columns)-1} features")
            
            # ========================================
            # 🔬 HYPERPARAMETER TUNING + PHASE 5 CACHING
            # ========================================
            best_params = None
            
            if enable_tuning and MODULES_AVAILABLE and model_name not in ['Linear Regression']:
                status_text.text(f"🔬 Optimizing {model_name}... ({i+1}/{total_models})")
                
                try:
                    # ⚡ PHASE 5: Check if this model was already optimized
                    if StateManager.is_optimization_completed(model_name):
                        cached_opt = StateManager.get_optimization_results(model_name)
                        
                        if cached_opt and 'best_params' in cached_opt:
                            best_params = cached_opt['best_params']
                            optimization_results[model_name] = cached_opt
                            
                            st.success(f"⚡ Using cached optimization for {model_name}")
                            meta = StateManager.get_optimization_metadata(model_name) or {}
                            if meta.get('n_trials'):
                                st.caption(f"Cached from {meta.get('n_trials')} trials, best RMSE: {meta.get('best_value'):.4f} (no re-optimization)")
                        else:
                            st.warning(f"⚠️ Optimization marked completed for {model_name} but cache empty. Re-running optimization.")
                            StateManager.clear_optimization_cache()
                    
                    # If no cached optimization, run it
                    if best_params is None:
                        optimizer = HyperparameterOptimizer(n_trials=n_trials, cv_splits=cv_splits)
                        X_train = train_df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
                        y_train = train_df[target_col]
                        
                        tuning_result = optimizer.optimize_model(
                            model_name.lower().replace(' ', '_'), 
                            X_train, y_train, metric='rmse'
                        )
                        
                        if 'error' not in tuning_result:
                            best_params = tuning_result['best_params']
                            optimization_results[model_name] = tuning_result
                            
                            # ✅ PHASE 5: Cache optimization for future runs
                            StateManager.set_optimization_results(model_name, tuning_result)
                            
                            st.info(f"✅ Hyperparameter optimization completed for {model_name}")
                            st.success("💾 Optimization results cached (will be reused next time)")
                        else:
                            st.warning(f"⚠️ Optimization error for {model_name}: {tuning_result.get('error')}")
                
                except Exception as e:
                    st.warning(f"⚠️ Hyperparameter tuning failed for {model_name}: {str(e)}")
            
            elif enable_tuning and model_name in ['Linear Regression']:
                st.info(f"ℹ️ {model_name} uses default parameters (optimization not applicable)")
            
            # ========================================
            # TRAIN MODEL
            # ========================================
            model, results = train_single_model(
                trainer, model_name, train_df, test_df, date_col or 'index', target_col, best_params
            )
            
            if model and results:
                # Mark optimization
                if best_params:
                    results['optimized'] = True
                    results['best_params'] = best_params
                else:
                    results['optimized'] = False
                
                # ✅ NEW: Store feature metadata
                results['num_features'] = len(train_df.columns) - 1  # Exclude target
                results['used_selected_features'] = bool(selected_features_list)
                if selected_features_list:
                    results['num_selected_features'] = len(selected_features_list)
                
                # Store results
                trained_models[model_name] = model
                model_results[model_name] = results
                trained_count += 1
                
                # ✅ Auto-save model
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    version = f"v_{timestamp}"
                    
                    model_path = trainer.save_model(model, model_name, version=version)
                    results['model_path'] = model_path
                    results['saved_at'] = datetime.now().isoformat()
                    
                    st.success(f"💾 {model_name} saved to: `{Path(model_path).name}`")
                except Exception as e:
                    st.warning(f"⚠️ Could not auto-save {model_name}: {str(e)}")
                
                # Backtesting
                if enable_backtesting and MODULES_AVAILABLE:
                    backtest_single_model(
                        model_name, model, df_for_training, target_col, date_col, 
                        initial_train_size, test_size_backtest, step_size, window_type,
                        backtest_results
                    )
                
                # Show results
                with results_container:
                    show_model_result(model_name, results)
            
            progress = (i + 1) / total_models
            progress_bar.progress(progress)
            time.sleep(0.3)
        
        except Exception as e:
            st.error(f"❌ {model_name} failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            continue
    
    # Save to StateManager
    StateManager.update({
        'trained_models': trained_models,
        'model_results': model_results,
        'backtest_results': backtest_results,
        'optimization_results': optimization_results
    })
    
    # Summary
    progress_bar.empty()
    status_text.empty()
    
    if trained_count > 0:
        st.success(f"🎉 Training completed! {trained_count}/{total_models} models trained successfully")
        
        # ✅ Show feature selection impact
        if selected_features_list:
            st.success(f"✨ Feature Selection Impact: Used {len(selected_features_list)} optimized features (from {len(df.columns)} total)")
            st.info("💡 Selected features chosen for best performance across all data groups")
        
        if enable_backtesting:
            st.success(f"📊 Backtesting completed for {len(backtest_results)} models")
        if enable_tuning:
            st.success(f"🔬 Hyperparameter tuning completed for {len(optimization_results)} models")
        st.balloons()
        
        show_training_summary()
    else:
        st.error("❌ No models were successfully trained")




def train_single_model(trainer, model_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                      date_col: str, target_col: str, best_params: dict = None):
    """Train a single model with error handling and optional hyperparameters."""
    try:
        if model_name == 'XGBoost':
            return trainer.train_xgboost(train_df, test_df, date_col, target_col, best_params)
        elif model_name == 'LightGBM':
            return trainer.train_lightgbm(train_df, test_df, date_col, target_col, best_params)
        elif model_name == 'Random Forest':
            return trainer.train_random_forest(train_df, test_df, date_col, target_col, best_params)
        elif model_name == 'CatBoost':
            return trainer.train_catboost(train_df, test_df, date_col, target_col, best_params)
        elif model_name == 'Linear Regression':
            return trainer.train_linear_regression(train_df, test_df, date_col, target_col, best_params)
        elif model_name == 'Ridge Regression':
            return trainer.train_ridge(train_df, test_df, date_col, target_col, best_params)
        elif model_name == 'Lasso Regression':
            return trainer.train_lasso(train_df, test_df, date_col, target_col, best_params)
        elif model_name == 'Decision Tree':
            return trainer.train_decision_tree(train_df, test_df, date_col, target_col, best_params)
        elif model_name == 'K-Nearest Neighbors':
            return trainer.train_knn(train_df, test_df, date_col, target_col, best_params)
        else:
            st.error(f"Unknown model: {model_name}")
            return None, None
    except Exception as e:
        st.error(f"Error training {model_name}: {str(e)}")
        return None, None


def backtest_single_model(model_name: str, model, df: pd.DataFrame, target_col: str, 
                         date_col: str, initial_train_size: int, test_size_backtest: int, 
                         step_size: int, window_type: str, backtest_results: dict):
    """Run backtesting for a single trained model."""
    try:
        backtester = Backtester(
            initial_train_size=initial_train_size,
            test_size=test_size_backtest,
            step_size=step_size,
            window_type=window_type
        )
        
        if date_col and date_col in df.columns:
            df_sorted = df.sort_values(date_col).reset_index(drop=True)
            date_series = df_sorted[date_col]
        else:
            df_sorted = df.reset_index(drop=True)
            date_series = None
        
        X = df_sorted.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
        y = df_sorted[target_col]
        
        backtest_result = backtester.backtest_model(
            model=model,
            X=X,
            y=y,
            date_col=date_series,
            refit=True
        )
        
        if 'error' not in backtest_result:
            backtest_results[model_name] = backtest_result
        
    except Exception as e:
        st.warning(f"Backtesting failed for {model_name}: {str(e)}")


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
        
        if results.get('optimized', False):
            st.success("✨ Model trained with optimized hyperparameters")
        
        if 'feature_importance' in results and results['feature_importance']:
            st.markdown("**🔍 Feature Importance**")
            importance_df = pd.DataFrame({
                'feature': list(results['feature_importance'].keys()),
                'importance': list(results['feature_importance'].values())
            }).sort_values('importance', ascending=False)
            
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
    
    model_results = StateManager.get('model_results', {})
    
    if not model_results:
        st.warning("No model results available")
        return
    
    comparison_data = []
    for model_name, results in model_results.items():
        rmse = results.get('test_rmse', results.get('rmse', np.nan))
        mae = results.get('test_mae', results.get('mae', np.nan))
        r2 = results.get('test_r2', results.get('r2', np.nan))
        training_time = results.get('training_time', 0)
        optimized = '✅' if results.get('optimized', False) else '❌'
        
        comparison_data.append({
            'Model': model_name,
            'Optimized': optimized,
            'Training Time (s)': training_time,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    st.markdown("#### 📊 MODEL PERFORMANCE COMPARISON")
    st.dataframe(comparison_df, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'RMSE' in comparison_df.columns and not comparison_df['RMSE'].isna().all():
            valid_rmse = comparison_df.dropna(subset=['RMSE'])
            if not valid_rmse.empty:
                fig = px.bar(valid_rmse.sort_values('RMSE'), 
                            x='Model', y='RMSE',
                            title="RMSE Comparison (Lower is Better)",
                            color='RMSE',
                            color_continuous_scale='Viridis')
                fig.update_layout(plot_bgcolor='#1a1d29', paper_bgcolor='#1a1d29',
                                font=dict(color='white'))
                display_plotly_chart(fig)
    
    with col2:
        if 'R²' in comparison_df.columns and not comparison_df['R²'].isna().all():
            valid_r2 = comparison_df.dropna(subset=['R²'])
            if not valid_r2.empty:
                fig = px.bar(valid_r2.sort_values('R²', ascending=False), 
                            x='Model', y='R²',
                            title="R² Comparison (Higher is Better)",
                            color='R²',
                            color_continuous_scale='Plasma')
                fig.update_layout(plot_bgcolor='#1a1d29', paper_bgcolor='#1a1d29',
                                font=dict(color='white'))
                display_plotly_chart(fig)
    
    if 'RMSE' in comparison_df.columns and not comparison_df['RMSE'].isna().all():
        valid_comparison = comparison_df.dropna(subset=['RMSE'])
        if not valid_comparison.empty:
            best_model_idx = valid_comparison['RMSE'].idxmin()
            best_model = valid_comparison.loc[best_model_idx, 'Model']
            StateManager.set('best_model_name', best_model)
            
            st.success(f"🏆 **BEST PERFORMING MODEL**: {best_model}")


def show_model_management():
    """
    Display model management interface for saving/loading models.
    
    TASK 3: Phase 2 - Model Persistence
    """
    st.markdown("### 💾 MODEL MANAGEMENT")
    
    config = get_config()
    model_registry_path = Path(config.model.model_registry_path)
    
    tab1, tab2, tab3 = st.tabs(["📁 Saved Models", "💾 Save Current Models", "📥 Load Model"])
    
    # TAB 1: View Saved Models
    with tab1:
        st.markdown("#### 📁 SAVED MODELS LIBRARY")
        
        if not model_registry_path.exists():
            st.info("No saved models yet. Train and save models to build your library.")
        else:
            saved_models = list(model_registry_path.glob("*.pkl"))
            
            if not saved_models:
                st.info("No saved models found in registry.")
            else:
                st.success(f"Found {len(saved_models)} saved models")
                
                model_data = []
                for model_file in saved_models:
                    file_stat = model_file.stat()
                    model_data.append({
                        'Filename': model_file.name,
                        'Size (KB)': f"{file_stat.st_size / 1024:.2f}",
                        'Modified': datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                        'Path': str(model_file)
                    })
                
                saved_df = pd.DataFrame(model_data)
                st.dataframe(saved_df, use_container_width=True)
                
                st.markdown("##### 🗑️ Delete Saved Model")
                model_to_delete = st.selectbox(
                    "Select model to delete",
                    options=[m['Filename'] for m in model_data],
                    key='delete_selector'
                )
                
                col_del1, col_del2 = st.columns([1, 3])
                with col_del1:
                    if st.button("🗑️ DELETE", type="secondary"):
                        try:
                            model_path = model_registry_path / model_to_delete
                            model_path.unlink()
                            st.success(f"✅ Deleted: {model_to_delete}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error deleting model: {str(e)}")
                
                with col_del2:
                    st.warning("⚠️ Deletion is permanent and cannot be undone!")
    
    # TAB 2: Save Current Models
    with tab2:
        st.markdown("#### 💾 SAVE CURRENTLY TRAINED MODELS")
        
        trained_models = StateManager.get('trained_models', {})
        model_results = StateManager.get('model_results', {})
        
        if not trained_models:
            st.info("No models currently trained. Train models first, then save them here.")
        else:
            st.success(f"{len(trained_models)} models available to save")
            
            models_to_save = st.multiselect(
                "Select models to save",
                options=list(trained_models.keys()),
                default=list(trained_models.keys()),
                key='save_selector'
            )
            
            col_save1, col_save2 = st.columns(2)
            
            with col_save1:
                version_tag = st.text_input(
                    "Version Tag (optional)",
                    placeholder="e.g., production, experiment_1",
                    help="Add a custom version tag to identify this model"
                )
            
            with col_save2:
                include_timestamp = st.checkbox("Include Timestamp", value=True)
            
            if st.button("💾 SAVE SELECTED MODELS", type="primary", use_container_width=True):
                if not models_to_save:
                    st.warning("Please select at least one model to save")
                else:
                    saved_count = 0
                    trainer = get_model_trainer()
                    
                    with st.spinner("Saving models..."):
                        for model_name in models_to_save:
                            try:
                                model = trained_models[model_name]
                                
                                if include_timestamp:
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    version = f"{version_tag}_{timestamp}" if version_tag else timestamp
                                else:
                                    version = version_tag if version_tag else "v1"
                                
                                model_path = trainer.save_model(model, model_name, version=version)
                                
                                metadata = {
                                    'model_name': model_name,
                                    'version': version,
                                    'saved_at': datetime.now().isoformat(),
                                    'metrics': model_results.get(model_name, {})
                                }
                                
                                metadata_path = Path(model_path).with_suffix('.json')
                                with open(metadata_path, 'w') as f:
                                    json.dump(metadata, f, indent=2, default=str)
                                
                                saved_count += 1
                                st.success(f"✅ Saved: {model_name} → `{Path(model_path).name}`")
                                
                            except Exception as e:
                                st.error(f"❌ Failed to save {model_name}: {str(e)}")
                    
                    if saved_count > 0:
                        st.balloons()
                        st.success(f"🎉 Successfully saved {saved_count}/{len(models_to_save)} models!")
    
    # TAB 3: Load Model
    with tab3:
        st.markdown("#### 📥 LOAD SAVED MODEL")
        
        if not model_registry_path.exists() or not list(model_registry_path.glob("*.pkl")):
            st.info("No saved models available to load.")
        else:
            saved_models = list(model_registry_path.glob("*.pkl"))
            model_files = {m.name: str(m) for m in saved_models}
            
            selected_file = st.selectbox(
                "Select model to load",
                options=list(model_files.keys()),
                key='load_selector'
            )
            
            if selected_file:
                metadata_path = Path(model_files[selected_file]).with_suffix('.json')
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        st.markdown("**📊 Model Information:**")
                        col_meta1, col_meta2, col_meta3 = st.columns(3)
                        
                        with col_meta1:
                            st.metric("Model", metadata.get('model_name', 'Unknown'))
                        with col_meta2:
                            st.metric("Version", metadata.get('version', 'N/A'))
                        with col_meta3:
                            saved_date = metadata.get('saved_at', '')
                            if saved_date:
                                saved_date = datetime.fromisoformat(saved_date).strftime('%Y-%m-%d %H:%M')
                            st.metric("Saved", saved_date)
                        
                        if 'metrics' in metadata and metadata['metrics']:
                            metrics = metadata['metrics']
                            st.markdown("**📈 Performance Metrics:**")
                            col_m1, col_m2, col_m3 = st.columns(3)
                            
                            with col_m1:
                                rmse = metrics.get('test_rmse', metrics.get('rmse', 'N/A'))
                                st.metric("RMSE", f"{rmse:.4f}" if isinstance(rmse, (int, float)) else rmse)
                            with col_m2:
                                mae = metrics.get('test_mae', metrics.get('mae', 'N/A'))
                                st.metric("MAE", f"{mae:.4f}" if isinstance(mae, (int, float)) else mae)
                            with col_m3:
                                r2 = metrics.get('test_r2', metrics.get('r2', 'N/A'))
                                st.metric("R²", f"{r2:.4f}" if isinstance(r2, (int, float)) else r2)
                    
                    except Exception as e:
                        st.warning(f"Could not load metadata: {str(e)}")
                
                if st.button("📥 LOAD THIS MODEL", type="primary", use_container_width=True):
                    try:
                        trainer = get_model_trainer()
                        loaded_model = trainer.load_model(model_files[selected_file])
                        
                        model_name = selected_file.split('_v_')[0].replace('_', ' ')
                        
                        trained_models = StateManager.get('trained_models', {})
                        trained_models[model_name] = loaded_model
                        StateManager.set('trained_models', trained_models)
                        
                        st.success(f"✅ Successfully loaded: {model_name}")
                        st.info("Model is now available in the current session for predictions!")
                        
                    except Exception as e:
                        st.error(f"❌ Failed to load model: {str(e)}")


if __name__ == "__main__":
    main()
