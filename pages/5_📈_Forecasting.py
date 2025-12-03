"""
Advanced Forecasting Page for CortexX Enterprise Platform
PHASE 3 - SESSION 10: Complete Interactive Forecasting Interface
✅ FINAL: Bulletproof with complete error handling
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
import json
import traceback

# Add scipy for confidence intervals
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    st.error("scipy not installed. Please run: pip install scipy")
    SCIPY_AVAILABLE = False

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# Imports
try:
    from src.utils.state_manager import StateManager, is_data_loaded, get_current_data
    from src.models.training import ModelTrainer
    from src.visualization.forecast_ui import ForecastVisualizer
    MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"Module import error: {e}")
    MODULES_AVAILABLE = False

st.set_page_config(
    page_title="Forecasting - CortexX",
    page_icon="📈",
    layout="wide"
)


def debug_forecast_state():
    """Debug function to check forecast state."""
    if 'forecast_results' in st.session_state:
        fr = st.session_state['forecast_results']
        st.warning("🔍 DEBUG MODE - Forecast State Analysis")
        
        with st.expander("Current Forecast State Details"):
            st.write("Type of forecast_results:", type(fr))
            st.write("Value:", fr)
            
            if isinstance(fr, dict):
                st.write("All keys present:", list(fr.keys()))
                
                for key in ['predictions', 'future_dates', 'lower_bound', 'upper_bound', 
                           'confidence', 'model_name', 'date_col', 'value_col', 'historical_df']:
                    if key in fr:
                        val = fr[key]
                        st.write(f"**{key}**: {type(val)}")
                        
                        if isinstance(val, np.ndarray):
                            st.write(f"  Shape: {val.shape}")
                            if len(val) > 0:
                                st.write(f"  First 3 values: {val[:3]}")
                            else:
                                st.error(f"  EMPTY ARRAY!")
                        
                        elif isinstance(val, pd.DatetimeIndex):
                            st.write(f"  Length: {len(val)}")
                            if len(val) > 0:
                                st.write(f"  First 3 dates: {val[:3]}")
                            else:
                                st.error(f"  EMPTY DATETIME INDEX!")
                        
                        elif isinstance(val, pd.DataFrame):
                            st.write(f"  DataFrame shape: {val.shape}")
                            st.write(f"  Columns: {list(val.columns)}")
                        
                        elif isinstance(val, (int, float, str)):
                            st.write(f"  Value: {val}")
                        
                        elif val is None:
                            st.error(f"  VALUE IS NONE!")
                        
                    else:
                        st.error(f"  ❌ MISSING KEY: {key}")
            else:
                st.error(f"❌ forecast_results is not a dictionary! It's a {type(fr)}")


def make_predictions_simple(model, historical_df, future_dates, date_col, value_col):
    """Make predictions using available features with robust error handling."""
    try:
        # Validate inputs
        if historical_df is None or historical_df.empty:
            st.error("❌ Historical DataFrame is empty or None!")
            return None
        
        if date_col not in historical_df.columns:
            st.error(f"❌ Date column '{date_col}' not found in DataFrame!")
            return None
        
        if value_col not in historical_df.columns:
            st.error(f"❌ Value column '{value_col}' not found in DataFrame!")
            return None
        
        if future_dates is None or len(future_dates) == 0:
            st.error("❌ Future dates are empty!")
            return None
        
        # Get the last value as fallback
        last_value = float(historical_df[value_col].iloc[-1]) if len(historical_df) > 0 else 0
        mean_value = float(historical_df[value_col].mean()) if len(historical_df) > 0 else 0
        
        # Try to use the model if it exists and has predict method
        if model is not None and hasattr(model, 'predict'):
            try:
                # Create future DataFrame with features
                future_df = pd.DataFrame({
                    date_col: future_dates
                })
                
                # Add date features that models often expect
                future_df['year'] = future_df[date_col].dt.year
                future_df['month'] = future_df[date_col].dt.month
                future_df['day'] = future_df[date_col].dt.day
                future_df['dayofweek'] = future_df[date_col].dt.dayofweek
                future_df['quarter'] = future_df[date_col].dt.quarter
                
                # Add lag features (simple approach)
                for lag in [1, 7, 30]:
                    if len(historical_df) >= lag:
                        future_df[f'lag_{lag}'] = historical_df[value_col].iloc[-lag]
                    else:
                        future_df[f'lag_{lag}'] = last_value
                
                # Try to predict
                predictions = model.predict(future_df)
                
                if predictions is not None and len(predictions) == len(future_dates):
                    st.success(f"✅ Model prediction successful!")
                    return np.array(predictions, dtype=float)
                else:
                    st.warning("⚠️ Model prediction returned unexpected format, using fallback")
            
            except Exception as model_error:
                st.warning(f"⚠️ Model prediction failed: {str(model_error)[:100]}")
        
        # FALLBACK: Use simple forecasting methods
        # Method 1: Simple moving average
        window_size = min(30, len(historical_df))
        if window_size > 0:
            moving_avg = historical_df[value_col].tail(window_size).mean()
        else:
            moving_avg = last_value
        
        # Method 2: Linear extrapolation (if we have enough data)
        predictions = []
        if len(historical_df) >= 10:
            # Calculate simple trend
            recent_data = historical_df[value_col].tail(10).values
            x = np.arange(len(recent_data))
            coeffs = np.polyfit(x, recent_data, 1)
            trend = coeffs[0]
            
            for i in range(len(future_dates)):
                pred = last_value + (trend * (i + 1))
                predictions.append(pred)
        else:
            # Just use moving average
            predictions = [moving_avg] * len(future_dates)
        
        return np.array(predictions, dtype=float)
        
    except Exception as e:
        st.error(f"❌ Prediction function failed: {str(e)}")
        
        # Ultimate fallback: return array of mean values
        if historical_df is not None and not historical_df.empty and value_col in historical_df.columns:
            mean_val = float(historical_df[value_col].mean())
            return np.full(len(future_dates), mean_val, dtype=float)
        else:
            return np.full(len(future_dates), 0.0, dtype=float)


def calculate_confidence_intervals(historical_values, predictions, confidence=0.95):
    """Calculate confidence intervals with robust error handling."""
    try:
        if historical_values is None or len(historical_values) == 0:
            std = np.std(predictions) if len(predictions) > 1 else predictions[0] * 0.1
        else:
            std = np.std(historical_values)
        
        if not SCIPY_AVAILABLE:
            # Simple fallback if scipy not available
            z_score = 1.96 if confidence == 0.95 else 2.58 if confidence == 0.99 else 1.645
        else:
            z_score = stats.norm.ppf((1 + confidence) / 2)
        
        margin = z_score * std
        lower = predictions - margin
        upper = predictions + margin
        
        # Ensure bounds are reasonable
        lower = np.maximum(lower, 0)  # Don't go below 0
        upper = np.maximum(upper, lower + 0.001)  # Ensure upper > lower
        
        return lower, upper
        
    except Exception as e:
        st.error(f"❌ Confidence interval calculation failed: {str(e)}")
        # Return wide intervals as fallback
        lower = predictions * 0.5
        upper = predictions * 1.5
        return lower, upper


def validate_forecast_results(results_dict):
    """Validate that all required forecast components exist."""
    # First check if it's actually a dictionary
    if not isinstance(results_dict, dict):
        return [f"Expected dict but got {type(results_dict)}"]
    
    required_keys = [
        'predictions', 'future_dates', 'lower_bound', 'upper_bound',
        'confidence', 'model_name', 'date_col', 'value_col', 'historical_df'
    ]
    
    missing_keys = []
    for key in required_keys:
        if key not in results_dict:
            missing_keys.append(key)
        elif results_dict[key] is None:
            missing_keys.append(f"{key} (None)")
    
    return missing_keys


def main():
    """Main forecasting function."""
    # DEBUG: Check initial state
    if 'forecast_results' in st.session_state:
        fr = st.session_state['forecast_results']
        if not isinstance(fr, dict):
            st.warning(f"⚠️ DEBUG: forecast_results initialized as {type(fr)} instead of dict")
            # Fix it immediately
            st.session_state['forecast_results'] = {}
            st.rerun()
    
    StateManager.initialize()
    
    st.markdown('''
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3rem; font-weight: 800; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            📈 Advanced Forecasting
        </h1>
        <p style="font-size: 1.2rem; color: #888;">
            Generate forecasts using your trained models
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    if not MODULES_AVAILABLE:
        st.error("❌ Required modules not available")
        return
    
    if not SCIPY_AVAILABLE:
        st.warning("⚠️ scipy not installed. Confidence intervals may be approximate.")
    
    if not is_data_loaded():
        st.warning("⚠️ Please load data first")
        return
    
    df = get_current_data()
    date_col = StateManager.get('date_column')
    value_col = StateManager.get('value_column')
    
    if not date_col or not value_col:
        st.warning("⚠️ Columns not detected")
        return
    
    st.info(f"📁 Data loaded: {len(df)} rows, Date column: '{date_col}', Value column: '{value_col}'")
    
    st.markdown("### 🤖 Select Trained Model")
    
    trained_models = StateManager.get('trained_models', {})
    model_results = StateManager.get('model_results', {})
    
    if not trained_models:
        st.warning("⚠️ No trained models found.")
        if st.button("🔄 Go to Training"):
            st.switch_page("pages/4_🤖_Model_Training.py")
        return
    
    st.success(f"✅ {len(trained_models)} model(s) found")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_model = st.selectbox("Model", list(trained_models.keys()))
    with col2:
        horizon = st.number_input("Days to forecast", 1, 365, 30)
    with col3:
        confidence = st.slider("Confidence %", 80, 99, 95)
    
    model = trained_models[selected_model]
    results = model_results.get(selected_model, {})
    
    with st.expander("📋 Model Info"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("RMSE", f"{results.get('test_rmse', 0):.2f}")
        with col_b:
            st.metric("MAE", f"{results.get('test_mae', 0):.2f}")
        with col_c:
            st.metric("R²", f"{results.get('test_r2', 0):.4f}")
    
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    
    with col_btn1:
        generate = st.button("🚀 Generate Forecast", type="primary", use_container_width=True)
    with col_btn2:
        compare = st.button("📊 Compare Models", use_container_width=True)
    with col_btn3:
        debug = st.button("🐛 Debug State", use_container_width=True)
    with col_btn4:
        if st.button("🔄 Clear Forecast", use_container_width=True):
            if 'forecast_results' in st.session_state:
                del st.session_state['forecast_results']
                st.success("Forecast cleared!")
                st.rerun()
    
    # ========================================================================
    # DEBUG MODE
    # ========================================================================
    if debug:
        debug_forecast_state()
        if 'forecast_results' in st.session_state:
            fr = st.session_state['forecast_results']
            missing = validate_forecast_results(fr)
            if missing:
                st.error(f"❌ Missing keys: {missing}")
            else:
                st.success("✅ All keys present!")
    
    # ========================================================================
    # GENERATE FORECAST
    # ========================================================================
    
    if generate:
        with st.spinner(f"🔮 Generating {horizon}-day forecast..."):
            progress_bar = st.progress(0)
            
            try:
                # Step 1: Validate data
                progress_bar.progress(10)
                
                if df is None or df.empty:
                    st.error("❌ No data available!")
                    return
                
                if date_col not in df.columns:
                    st.error(f"❌ Date column '{date_col}' not found!")
                    return
                
                if value_col not in df.columns:
                    st.error(f"❌ Value column '{value_col}' not found!")
                    return
                
                # Ensure date column is datetime
                if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                    try:
                        df[date_col] = pd.to_datetime(df[date_col])
                    except:
                        st.error(f"❌ Cannot convert '{date_col}' to datetime!")
                        return
                
                # Step 2: Create future dates
                progress_bar.progress(30)
                
                last_date = df[date_col].max()
                future_dates = pd.date_range(
                    start=last_date + timedelta(days=1),
                    periods=horizon,
                    freq='D'
                )
                
                # Step 3: Make predictions
                progress_bar.progress(50)
                
                predictions = make_predictions_simple(model, df, future_dates, date_col, value_col)
                
                if predictions is None:
                    st.error("❌ Predictions returned None!")
                    return
                
                # Step 4: Calculate intervals
                progress_bar.progress(70)
                
                lower_bound, upper_bound = calculate_confidence_intervals(
                    df[value_col].values, predictions, confidence/100
                )
                
                # Step 5: Prepare results
                progress_bar.progress(90)
                
                results_dict = {
                    'model_name': selected_model,
                    'future_dates': future_dates,
                    'predictions': predictions,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'confidence': confidence,
                    'historical_df': df.copy(),
                    'date_col': date_col,
                    'value_col': value_col
                }
                
                # Validate results
                missing_keys = validate_forecast_results(results_dict)
                
                if missing_keys:
                    st.error(f"❌ Forecast incomplete! Missing: {missing_keys}")
                    return
                
                # Store in session state - IMPORTANT: as a DICTIONARY, not list
                st.session_state['forecast_results'] = results_dict
                progress_bar.progress(100)
                st.success("✅ Forecast generated successfully!")
                st.balloons()
                st.rerun()  # Force refresh to show results
                
            except Exception as e:
                st.error(f"❌ Error during forecast generation: {str(e)}")
                with st.expander("🔍 Detailed Error Traceback"):
                    st.code(traceback.format_exc())
    
    # ========================================================================
    # DISPLAY RESULTS
    # ========================================================================
    
    if 'forecast_results' in st.session_state:
        fr = st.session_state['forecast_results']
        
        # ✅ CRITICAL FIX: Check if fr is actually a dictionary
        if not isinstance(fr, dict):
            st.error(f"""
            ⚠️ **Data Type Error!**
            
            Expected: Dictionary with forecast results
            Got: {type(fr)} with value: {fr}
            
            This happens when forecast_results is not properly initialized.
            """)
            
            # Auto-fix: Convert to empty dictionary
            st.session_state['forecast_results'] = {}
            st.rerun()
            return
        
        # Check if dictionary is empty
        if len(fr) == 0:
            st.info("📭 Forecast results dictionary is empty. Generate a forecast first!")
            return
        
        # Validate that all required keys exist
        required_keys = [
            'predictions', 'future_dates', 'lower_bound', 'upper_bound',
            'confidence', 'model_name', 'date_col', 'value_col', 'historical_df'
        ]
        
        missing_keys = []
        for key in required_keys:
            if key not in fr:
                missing_keys.append(key)
            elif fr[key] is None:
                missing_keys.append(f"{key} (is None)")
        
        if missing_keys:
            st.error(f"⚠️ Incomplete forecast. Missing: {missing_keys}")
            with st.expander("🔍 Debug Details"):
                st.write("Current forecast results keys:", list(fr.keys()))
                st.write("Expected keys:", required_keys)
            
            if st.button("🗑️ Clear Invalid Forecast"):
                del st.session_state['forecast_results']
                st.rerun()
        else:
            # ✅ ALL KEYS EXIST - SAFE TO DISPLAY
            st.markdown("---")
            st.markdown(f"### 📊 Forecast Results: {fr['model_name']}")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Forecast Horizon", f"{len(fr['predictions'])} days")
            with col2:
                st.metric("Average Forecast", f"{np.mean(fr['predictions']):,.0f}")
            with col3:
                st.metric("Total Forecast", f"{np.sum(fr['predictions']):,.0f}")
            with col4:
                st.metric("Confidence Level", f"{fr['confidence']}%")
            
            st.markdown("---")
            
            # Chart
            st.markdown("#### 📈 Visualization")
            
            try:
                # Create forecast DataFrame
                forecast_df = pd.DataFrame({
                    fr['date_col']: fr['future_dates'],
                    'forecast': fr['predictions'],
                    'lower_bound': fr['lower_bound'],
                    'upper_bound': fr['upper_bound']
                })
                
                # Try to use the visualizer if available
                try:
                    visualizer = ForecastVisualizer()
                    fig = visualizer.create_forecast_plot(
                        historical_df=fr['historical_df'],
                        forecast_df=forecast_df,
                        date_col=fr['date_col'],
                        value_col=fr['value_col'],
                        forecast_col='forecast',
                        lower_bound_col='lower_bound',
                        upper_bound_col='upper_bound',
                        title=f"{fr['model_name']} - {len(fr['predictions'])}-Day Forecast"
                    )
                except:
                    # Fallback visualization
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=fr['historical_df'][fr['date_col']],
                        y=fr['historical_df'][fr['value_col']],
                        mode='lines',
                        name='Historical',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Forecast
                    fig.add_trace(go.Scatter(
                        x=fr['future_dates'],
                        y=fr['predictions'],
                        mode='lines',
                        name='Forecast',
                        line=dict(color='red', width=3, dash='dash')
                    ))
                    
                    # Confidence interval
                    fig.add_trace(go.Scatter(
                        x=list(fr['future_dates']) + list(fr['future_dates'])[::-1],
                        y=list(fr['upper_bound']) + list(fr['lower_bound'])[::-1],
                        fill='toself',
                        fillcolor='rgba(255, 0, 0, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name=f'{fr["confidence"]}% Confidence'
                    ))
                    
                    fig.update_layout(
                        title=f"{fr['model_name']} Forecast",
                        xaxis_title="Date",
                        yaxis_title=fr['value_col'],
                        hovermode='x unified',
                        showlegend=True
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Chart error: {e}")
                with st.expander("Chart Error Details"):
                    st.code(traceback.format_exc())
            
            st.markdown("---")
            
            # Table
            st.markdown("#### 📋 Forecast Data")
            
            table_df = pd.DataFrame({
                'Date': fr['future_dates'],
                'Forecast': fr['predictions'],
                f'Lower {fr["confidence"]}%': fr['lower_bound'],
                f'Upper {fr["confidence"]}%': fr['upper_bound']
            })
            
            # Format numbers
            table_df['Forecast'] = table_df['Forecast'].map('{:,.2f}'.format)
            table_df[f'Lower {fr["confidence"]}%'] = table_df[f'Lower {fr["confidence"]}%'].map('{:,.2f}'.format)
            table_df[f'Upper {fr["confidence"]}%'] = table_df[f'Upper {fr["confidence"]}%'].map('{:,.2f}'.format)
            
            st.dataframe(table_df, use_container_width=True, height=400)
            
            st.markdown("---")
            
            # Export
            st.markdown("#### 💾 Export Forecast")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV Export
                csv = table_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📄 Download CSV",
                    csv,
                    f"forecast_{fr['model_name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                # JSON Export
                export_data = {
                    'model': fr['model_name'],
                    'generated_at': datetime.now().isoformat(),
                    'horizon': len(fr['predictions']),
                    'confidence': fr['confidence'],
                    'forecast': [
                        {
                            'date': d.strftime('%Y-%m-%d'),
                            'forecast': float(p),
                            'lower_bound': float(l),
                            'upper_bound': float(u)
                        }
                        for d, p, l, u in zip(fr['future_dates'], fr['predictions'], fr['lower_bound'], fr['upper_bound'])
                    ]
                }
                
                json_str = json.dumps(export_data, indent=2)
                st.download_button(
                    "📋 Download JSON",
                    json_str,
                    f"forecast_{fr['model_name'].replace(' ', '_')}.json",
                    "application/json",
                    use_container_width=True
                )
    
    # ========================================================================
    # MODEL COMPARISON
    # ========================================================================
    
    if compare and len(trained_models) > 1:
        st.markdown("---")
        st.markdown("### 🔬 Model Comparison")
        
        comp_data = []
        for name, model_obj in trained_models.items():
            if name in model_results:
                r = model_results[name]
                comp_data.append({
                    'Model': name,
                    'RMSE': r.get('test_rmse', 0),
                    'MAE': r.get('test_mae', 0),
                    'R²': r.get('test_r2', 0),
                    'Type': str(type(model_obj).__name__)
                })
        
        if comp_data:
            df_comp = pd.DataFrame(comp_data)
            
            # Sort by RMSE (lower is better)
            df_comp = df_comp.sort_values('RMSE')
            
            st.success(f"🏆 Best Model: **{df_comp.iloc[0]['Model']}** (RMSE: {df_comp.iloc[0]['RMSE']:.2f})")
            st.dataframe(df_comp, use_container_width=True)
            
            # Visualization
            fig = px.bar(df_comp, x='Model', y='RMSE', title='Model Performance (RMSE)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model results available for comparison.")
    elif compare:
        st.info("Need 2+ trained models to compare.")


if __name__ == "__main__":
    main()