"""
Professional Model Evaluation with Performance Analysis - PRODUCTION READY VERSION

PHASE 2 INTEGRATED:
- Uses StateManager for all state operations
- Cached visualizer singleton
- All advanced features preserved
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os
import io
import base64

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# ✅ PHASE 2 IMPORTS
try:
    from src.utils.state_manager import StateManager, is_data_loaded, get_current_data
    from src.models.evaluation import ModelEvaluator
    from src.models.backtesting import Backtester
    from src.visualization.dashboard import get_visualizer, display_plotly_chart
    from src.utils.config import get_config
    MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"Model evaluation modules not available: {e}")
    MODULES_AVAILABLE = False

st.set_page_config(
    page_title="Model Evaluation - CortexX",
    page_icon="📋",
    layout="wide"
)


def main():
    """Main model evaluation function."""
    
    st.markdown('<div class="section-header">📋 ENTERPRISE MODEL EVALUATION</div>', unsafe_allow_html=True)
    
    # ✅ UPDATED: Use StateManager
    StateManager.initialize()
    
    trained_models = StateManager.get('trained_models', {})
    
    if not trained_models:
        st.warning("⚠️ Please train models first from the Model Training page")
        return
    
    if not MODULES_AVAILABLE:
        st.error("❌ Model evaluation modules not available.")
        return
    
    # Initialize components
    evaluator = ModelEvaluator()
    # ✅ UPDATED: Use cached singleton
    visualizer = get_visualizer()
    
    # Model Evaluation Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Model Comparison", 
        "📊 Performance Analysis", 
        "🔄 Backtesting", 
        "🎯 Advanced Insights"
    ])
    
    with tab1:
        render_model_comparison(evaluator, visualizer)
    
    with tab2:
        render_performance_analysis(visualizer)
    
    with tab3:
        render_backtesting_analysis(visualizer)
    
    with tab4:
        render_advanced_insights()


def render_model_comparison(evaluator: ModelEvaluator, visualizer):
    """Render comprehensive model comparison."""
    
    st.markdown("""
    <div class="enterprise-card">
        <div class="card-title">🏆 ENTERPRISE MODEL COMPARISON</div>
        <div class="card-description">Comprehensive comparison of all trained models across multiple performance metrics with interactive visualizations and detailed analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    # ✅ UPDATED: Use StateManager
    model_results = StateManager.get('model_results', {})
    
    if not model_results:
        st.warning("No model results available for comparison")
        return
    
    # Create comparison dataframe
    comparison_df = evaluator.compare_models(model_results)
    
    if comparison_df.empty:
        st.warning("No valid comparison data available")
        return
    
    # Display comparison table
    st.markdown("#### 📊 PERFORMANCE METRICS COMPARISON")
    st.dataframe(comparison_df, use_container_width=True)
    
    # Interactive comparison charts
    st.markdown("#### 📈 VISUAL COMPARISON")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Metric selection
        metric = st.selectbox(
            "Select Metric for Comparison",
            ['RMSE', 'MAE', 'R2', 'MAPE', 'Training Time (s)'],
            help="Choose the metric to compare across models"
        )
        
        if metric in comparison_df.columns:
            # Bar chart comparison
            if metric in ['RMSE', 'MAE', 'MAPE']:
                sort_ascending = True  # Lower is better
            else:
                sort_ascending = False  # Higher is better
            
            sorted_df = comparison_df.sort_values(metric, ascending=sort_ascending)
            
            fig = px.bar(sorted_df, x='Model', y=metric,
                        title=f"Model Comparison - {metric}",
                        color=metric,
                        color_continuous_scale='Viridis')
            fig.update_layout(plot_bgcolor='#1a1d29', paper_bgcolor='#1a1d29',
                            font=dict(color='white'))
            display_plotly_chart(fig)
    
    with col2:
        # Radar chart for multiple metrics
        st.markdown("**🎯 MULTI-METRIC RADAR CHART**")
        
        # Select metrics for radar chart
        radar_metrics = st.multiselect(
            "Select metrics for radar analysis",
            ['RMSE', 'MAE', 'R2', 'MAPE'],
            default=['RMSE', 'MAE', 'R2']
        )
        
        if radar_metrics:
            try:
                fig = create_radar_chart(comparison_df, radar_metrics)
                display_plotly_chart(fig)
            except Exception as e:
                st.error(f"Error creating radar chart: {str(e)}")
    
    # Best model identification
    st.markdown("---")
    st.markdown("#### 🎯 BEST MODEL ANALYSIS")
    
    if 'RMSE' in comparison_df.columns:
        best_model_rmse = comparison_df.loc[comparison_df['RMSE'].idxmin(), 'Model']
        if 'R2' in comparison_df.columns:
            best_model_r2 = comparison_df.loc[comparison_df['R2'].idxmax(), 'Model']
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.success(f"🏆 **BEST BY RMSE**: {best_model_rmse}")
            st.metric("Lowest RMSE", f"{comparison_df['RMSE'].min():.4f}")
        
        with col_b:
            if 'R2' in comparison_df.columns:
                st.success(f"🏆 **BEST BY R²**: {best_model_r2}")
                st.metric("Highest R²", f"{comparison_df['R2'].max():.4f}")
        
        # ✅ UPDATED: Store using StateManager
        StateManager.set('best_model_name', best_model_rmse)


def create_radar_chart(comparison_df: pd.DataFrame, metrics: list) -> go.Figure:
    """Create a radar chart comparing models across multiple metrics."""
    
    # Normalize metrics for radar chart
    normalized_data = []
    models = comparison_df['Model'].tolist()
    
    for metric in metrics:
        if metric in comparison_df.columns:
            values = comparison_df[metric].values
            
            # Check for valid values
            if np.all(np.isnan(values)) or np.all(np.isinf(values)):
                continue
            
            # Normalize based on metric type
            values_min = np.nanmin(values)
            values_max = np.nanmax(values)
            
            if values_max == values_min:
                normalized = np.ones_like(values)
            elif metric in ['RMSE', 'MAE', 'MAPE']:
                # Lower is better - invert normalization
                normalized = 1 - (values - values_min) / (values_max - values_min)
            else:
                # Higher is better - direct normalization
                normalized = (values - values_min) / (values_max - values_min)
            
            normalized_data.append(normalized)
    
    if not normalized_data:
        return go.Figure()
    
    # Create radar chart
    fig = go.Figure()
    
    for i, model in enumerate(models):
        model_values = [data[i] for data in normalized_data]
        fig.add_trace(go.Scatterpolar(
            r=model_values + [model_values[0]],  # Close the polygon
            theta=metrics + [metrics[0]],
            fill='toself',
            name=model
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Multi-Metric Model Comparison Radar Chart",
        height=500,
        plot_bgcolor='#1a1d29',
        paper_bgcolor='#1a1d29',
        font=dict(color='white')
    )
    
    return fig


def render_performance_analysis(visualizer):
    """Render detailed performance analysis for individual models."""
    
    st.markdown("""
    <div class="enterprise-card">
        <div class="card-title">📊 DETAILED PERFORMANCE ANALYSIS</div>
        <div class="card-description">In-depth analysis of individual model performance with comprehensive visualizations, residual analysis, and feature importance examination</div>
    </div>
    """, unsafe_allow_html=True)
    
    # ✅ UPDATED: Use StateManager
    model_results = StateManager.get('model_results', {})
    
    if not model_results:
        st.warning("No model results available")
        return
    
    # Model selection for detailed analysis
    selected_model = st.selectbox(
        "Select Model for Detailed Analysis",
        list(model_results.keys()),
        key="detail_model"
    )
    
    if selected_model not in model_results:
        return
    
    results = model_results[selected_model]
    
    # Performance metrics
    st.markdown("#### 📈 PERFORMANCE METRICS")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        rmse_val = results.get('test_rmse', results.get('rmse', 0))
        st.metric("RMSE", f"{rmse_val:.4f}" if isinstance(rmse_val, (int, float)) and not np.isnan(rmse_val) else "N/A")
    with col2:
        mae_val = results.get('test_mae', results.get('mae', 0))
        st.metric("MAE", f"{mae_val:.4f}" if isinstance(mae_val, (int, float)) and not np.isnan(mae_val) else "N/A")
    with col3:
        r2_val = results.get('test_r2', results.get('r2', 0))
        st.metric("R²", f"{r2_val:.4f}" if isinstance(r2_val, (int, float)) and not np.isnan(r2_val) else "N/A")
    with col4:
        time_val = results.get('training_time', 0)
        st.metric("Training Time", f"{time_val:.2f}s" if isinstance(time_val, (int, float)) else "N/A")
    
    # Visualization tabs
    tab1, tab2, tab3 = st.tabs(["📈 Forecast vs Actual", "📉 Residual Analysis", "📊 Feature Importance"])
    
    with tab1:
        # Forecast vs Actual plot
        try:
            # Get actual and predicted values with proper fallbacks
            y_actual = results.get('y_test', results.get('actual', np.array([])))
            y_predicted = results.get('test_predictions', results.get('predictions', np.array([])))
            
            # Convert to numpy arrays if needed
            if not isinstance(y_actual, np.ndarray):
                y_actual = np.array(y_actual) if y_actual is not None else np.array([])
            if not isinstance(y_predicted, np.ndarray):
                y_predicted = np.array(y_predicted) if y_predicted is not None else np.array([])
            
            if len(y_actual) > 0 and len(y_predicted) > 0:
                # Ensure same length
                min_len = min(len(y_actual), len(y_predicted))
                y_actual = y_actual[:min_len]
                y_predicted = y_predicted[:min_len]
                
                # Create comparison plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=y_actual,
                    mode='lines',
                    name='Actual',
                    line=dict(color='#00d4ff', width=2)
                ))
                fig.add_trace(go.Scatter(
                    y=y_predicted,
                    mode='lines',
                    name='Predicted',
                    line=dict(color='#ff6b6b', width=2, dash='dash')
                ))
                fig.update_layout(
                    title=f"{selected_model} - Forecast vs Actual",
                    xaxis_title="Time",
                    yaxis_title="Value",
                    plot_bgcolor='#1a1d29',
                    paper_bgcolor='#1a1d29',
                    font=dict(color='white'),
                    height=500
                )
                display_plotly_chart(fig)
            else:
                st.info("No test data available for visualization")
                
        except Exception as e:
            st.error(f"Error creating forecast plot: {str(e)}")
    
    with tab2:
        # Residual analysis
        try:
            y_actual = results.get('y_test', results.get('actual', np.array([])))
            y_predicted = results.get('test_predictions', results.get('predictions', np.array([])))
            
            # Convert to numpy arrays if needed
            if not isinstance(y_actual, np.ndarray):
                y_actual = np.array(y_actual) if y_actual is not None else np.array([])
            if not isinstance(y_predicted, np.ndarray):
                y_predicted = np.array(y_predicted) if y_predicted is not None else np.array([])
            
            if len(y_actual) > 0 and len(y_predicted) > 0:
                # Ensure same length
                min_len = min(len(y_actual), len(y_predicted))
                y_actual = y_actual[:min_len]
                y_predicted = y_predicted[:min_len]
                
                residuals = y_actual - y_predicted
                
                fig = visualizer.create_residual_analysis_plot(y_actual, y_predicted)
                display_plotly_chart(fig)
                
                # Residual statistics
                st.markdown("#### 📊 RESIDUAL STATISTICS")
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("Mean Residual", f"{np.mean(residuals):.4f}")
                with col_b:
                    st.metric("Std Residual", f"{np.std(residuals):.4f}")
                with col_c:
                    st.metric("Min Residual", f"{np.min(residuals):.4f}")
                with col_d:
                    st.metric("Max Residual", f"{np.max(residuals):.4f}")
                    
            else:
                st.info("No test data available for residual analysis")
                
        except Exception as e:
            st.error(f"Error creating residual analysis: {str(e)}")
    
    with tab3:
        # Feature importance
        try:
            if 'feature_importance' in results and results['feature_importance']:
                importance_df = pd.DataFrame({
                    'feature': list(results['feature_importance'].keys()),
                    'importance': list(results['feature_importance'].values())
                }).sort_values('importance', ascending=True)
                
                fig = visualizer.create_feature_importance_plot(importance_df)
                display_plotly_chart(fig)
            else:
                st.info("No feature importance data available for this model")
                
        except Exception as e:
            st.error(f"Error creating feature importance plot: {str(e)}")


def render_backtesting_analysis(visualizer):
    """Render backtesting analysis."""
    
    st.markdown("""
    <div class="enterprise-card">
        <div class="card-title">🔄 BACKTESTING ANALYSIS</div>
        <div class="card-description">Comprehensive walk-forward validation results showing model performance across multiple time windows for robust time series evaluation</div>
    </div>
    """, unsafe_allow_html=True)
    
    # ✅ UPDATED: Use StateManager
    backtest_results = StateManager.get('backtest_results', {})
    
    if not backtest_results:
        st.info("No backtesting results available. Enable backtesting in the Model Training page.")
        return
    
    # Backtesting summary
    st.markdown("#### 📊 BACKTESTING PERFORMANCE SUMMARY")
    
    summary_data = []
    for model_name, results in backtest_results.items():
        if 'aggregate_metrics' in results:
            metrics = results['aggregate_metrics']
            summary_data.append({
                'Model': model_name,
                'Overall RMSE': f"{metrics.get('overall_rmse', 0):.4f}",
                'Avg RMSE': f"{metrics.get('avg_rmse', 0):.4f}",
                'Overall R²': f"{metrics.get('overall_r2', 0):.4f}",
                'Windows': results.get('n_windows', 0)
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Backtesting visualization
        selected_backtest_model = st.selectbox(
            "Select Model for Backtesting Details",
            list(backtest_results.keys()),
            key="backtest_model"
        )
        
        if selected_backtest_model in backtest_results:
            results = backtest_results[selected_backtest_model]
            
            # Check for iteration_results (correct key from Backtester class)
            if 'iteration_results' in results:
                window_metrics = results['iteration_results']
                windows = list(range(len(window_metrics)))
                rmse_values = [wm.get('rmse', 0) for wm in window_metrics]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=windows, y=rmse_values,
                    mode='lines+markers',
                    name='Window RMSE',
                    line=dict(color='#00d4ff', width=3),
                    marker=dict(size=6)
                ))
                fig.update_layout(
                    title=f"{selected_backtest_model} - RMSE Across Backtesting Windows",
                    xaxis_title="Window Number",
                    yaxis_title="RMSE",
                    plot_bgcolor='#1a1d29',
                    paper_bgcolor='#1a1d29',
                    font=dict(color='white'),
                    height=400
                )
                display_plotly_chart(fig)
            else:
                st.info("No window-level metrics available for detailed visualization")


def render_advanced_insights():
    """Render advanced insights and recommendations - FIXED VERSION."""
    
    st.markdown("""
    <div class="enterprise-card">
        <div class="card-title">🎯 ADVANCED INSIGHTS & RECOMMENDATIONS</div>
        <div class="card-description">Strategic insights and actionable recommendations based on comprehensive model evaluation to optimize your forecasting strategy</div>
    </div>
    """, unsafe_allow_html=True)
    
    # ✅ UPDATED: Use StateManager
    model_results = StateManager.get('model_results', {})
    
    if not model_results:
        st.warning("No model results available for insights")
        return
    
    # Overall insights
    st.markdown("#### 📈 STRATEGIC INSIGHTS")
    
    # Model recommendations
    try:
        # Find best model by RMSE with proper validation
        valid_models = {}
        for model_name, results in model_results.items():
            rmse = results.get('test_rmse', results.get('rmse', float('inf')))
            if rmse != float('inf') and not np.isnan(rmse):
                valid_models[model_name] = rmse
        
        if not valid_models:
            st.warning("No valid model results available for analysis")
            return
        
        best_model = min(valid_models.items(), key=lambda x: x[1])[0]
        
        # Performance categorization with FIXED percentile calculation
        good_models = []
        average_models = []
        
        rmse_values = list(valid_models.values())
        
        # CRITICAL FIX: Check if we have enough valid values before calculating percentiles
        if len(rmse_values) >= 4:  # Need at least 4 values for meaningful quartiles
            rmse_25 = np.percentile(rmse_values, 25)
            rmse_75 = np.percentile(rmse_values, 75)
            
            for model_name, rmse in valid_models.items():
                if rmse <= rmse_25:
                    good_models.append(model_name)
                elif rmse <= rmse_75:
                    average_models.append(model_name)
        elif len(rmse_values) > 0:
            # If we have fewer than 4 models, use median as threshold
            rmse_median = np.median(rmse_values)
            for model_name, rmse in valid_models.items():
                if rmse <= rmse_median:
                    good_models.append(model_name)
                else:
                    average_models.append(model_name)
        
        # Display recommendations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🏆 RECOMMENDED MODELS**")
            st.success(f"**Best Performer**: {best_model}")
            
            if good_models:
                st.markdown("**High Performers**:")
                for model in good_models:
                    if model != best_model:
                        st.write(f"• {model}")
            
            if len(valid_models) > 0:
                st.info(f"Total models evaluated: {len(valid_models)}")
        
        with col2:
            st.markdown("**💡 STRATEGIC RECOMMENDATIONS**")
            st.info("""
            - Use **ensemble methods** for robust production predictions
            - **Monitor performance** regularly with new incoming data
            - Consider **automatic retraining** when performance degrades
            - Use **confidence intervals** for risk assessment and planning
            - Implement **model versioning** for tracking improvements
            """)
        
        # Performance distribution
        if len(rmse_values) > 1:
            st.markdown("#### 📊 PERFORMANCE DISTRIBUTION")
            col_x, col_y, col_z = st.columns(3)
            
            with col_x:
                st.metric("Best RMSE", f"{min(rmse_values):.4f}")
            with col_y:
                st.metric("Average RMSE", f"{np.mean(rmse_values):.4f}")
            with col_z:
                st.metric("Worst RMSE", f"{max(rmse_values):.4f}")
    
    except Exception as e:
        st.error(f"Error generating insights: {str(e)}")
        st.info("Please ensure models are properly trained with valid results")
    
    # Enhanced export capabilities with multiple formats
    st.markdown("---")
    st.markdown("#### 💾 ENTERPRISE EXPORT CAPABILITIES")
    
    col_x, col_y, col_z = st.columns(3)
    
    with col_x:
        if st.button("📄 EXPORT COMPREHENSIVE REPORT", use_container_width=True):
            export_comprehensive_report()
    
    with col_y:
        if st.button("📊 EXPORT PERFORMANCE DATA", use_container_width=True):
            export_performance_data()
    
    with col_z:
        if st.button("🎯 EXPORT STRATEGIC INSIGHTS", use_container_width=True):
            export_strategic_insights()


def export_comprehensive_report():
    """Export comprehensive model evaluation report in multiple formats."""
    
    try:
        # ✅ UPDATED: Use StateManager
        model_results = StateManager.get('model_results', {})
        backtest_results = StateManager.get('backtest_results', {})
        
        # Create comprehensive report
        report_content = "# CORTEXX ENTERPRISE MODEL EVALUATION REPORT\n\n"
        report_content += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report_content += f"**Platform**: CortexX Enterprise Forecasting v3.0\n\n"
        
        # Add model comparison
        if model_results:
            report_content += "## 📊 MODEL PERFORMANCE SUMMARY\n\n"
            
            # Create comparison table
            comparison_data = []
            for model_name, results in model_results.items():
                rmse_val = results.get('test_rmse', results.get('rmse', 'N/A'))
                mae_val = results.get('test_mae', results.get('mae', 'N/A'))
                r2_val = results.get('test_r2', results.get('r2', 'N/A'))
                time_val = results.get('training_time', 'N/A')
                
                comparison_data.append({
                    'Model': model_name,
                    'RMSE': f"{rmse_val:.4f}" if isinstance(rmse_val, (int, float)) else rmse_val,
                    'MAE': f"{mae_val:.4f}" if isinstance(mae_val, (int, float)) else mae_val,
                    'R²': f"{r2_val:.4f}" if isinstance(r2_val, (int, float)) else r2_val,
                    'Training Time (s)': f"{time_val:.2f}" if isinstance(time_val, (int, float)) else time_val
                })
            
            # Convert to markdown table
            if comparison_data:
                headers = comparison_data[0].keys()
                report_content += "| " + " | ".join(headers) + " |\n"
                report_content += "|" + "|".join(["---"] * len(headers)) + "|\n"
                
                for row in comparison_data:
                    report_content += "| " + " | ".join(str(x) for x in row.values()) + " |\n"
        
        # Add backtesting results
        if backtest_results:
            report_content += "\n## 🔄 BACKTESTING RESULTS\n\n"
            for model_name, results in backtest_results.items():
                if 'aggregate_metrics' in results:
                    metrics = results['aggregate_metrics']
                    report_content += f"### {model_name}\n"
                    report_content += f"- Overall RMSE: {metrics.get('overall_rmse', 'N/A'):.4f}\n"
                    report_content += f"- Average RMSE: {metrics.get('avg_rmse', 'N/A'):.4f}\n"
                    report_content += f"- Windows: {results.get('n_windows', 'N/A')}\n\n"
        
        # Add strategic recommendations
        report_content += "\n## 🎯 STRATEGIC RECOMMENDATIONS\n\n"
        report_content += "1. **Production Deployment**: Consider the best performing model for production use\n"
        report_content += "2. **Monitoring**: Implement continuous performance monitoring\n"
        report_content += "3. **Retraining**: Schedule regular model retraining with new data\n"
        report_content += "4. **Ensemble Methods**: Explore ensemble approaches for improved robustness\n"
        report_content += "5. **Uncertainty Quantification**: Always use confidence intervals for decision making\n"
        
        # Download buttons for multiple formats
        st.markdown("### 📤 DOWNLOAD REPORT")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Markdown format
            st.download_button(
                label="📝 MARKDOWN REPORT",
                data=report_content,
                file_name=f"cortexx_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        with col2:
            # Text format
            st.download_button(
                label="📄 TEXT REPORT",
                data=report_content,
                file_name=f"cortexx_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col3:
            # HTML format
            html_content = report_content.replace('\n', '<br>').replace('##', '<h2>').replace('#', '<h1>')
            st.download_button(
                label="🌐 HTML REPORT",
                data=html_content,
                file_name=f"cortexx_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
                mime="text/html",
                use_container_width=True
            )
        
        st.success("✅ Report generated successfully!")
        
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")


def export_performance_data():
    """Export performance data in multiple formats."""
    
    try:
        # ✅ UPDATED: Use StateManager
        model_results = StateManager.get('model_results', {})
        
        # Create performance data
        performance_data = []
        for model_name, results in model_results.items():
            performance_data.append({
                'model': model_name,
                'rmse': results.get('test_rmse', results.get('rmse', np.nan)),
                'mae': results.get('test_mae', results.get('mae', np.nan)),
                'r2': results.get('test_r2', results.get('r2', np.nan)),
                'training_time': results.get('training_time', np.nan),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        performance_df = pd.DataFrame(performance_data)
        
        # Export options
        st.markdown("### 📊 DOWNLOAD PERFORMANCE DATA")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV format
            csv_data = performance_df.to_csv(index=False)
            st.download_button(
                label="📈 CSV DATA",
                data=csv_data,
                file_name=f"cortexx_performance_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # JSON format
            json_data = performance_df.to_json(orient='records', indent=2)
            st.download_button(
                label="📋 JSON DATA",
                data=json_data,
                file_name=f"cortexx_performance_data_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col3:
            # Excel format
            try:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    performance_df.to_excel(writer, index=False, sheet_name='Performance_Metrics')
                excel_data = output.getvalue()
                st.download_button(
                    label="📊 EXCEL DATA",
                    data=excel_data,
                    file_name=f"cortexx_performance_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.ms-excel",
                    use_container_width=True
                )
            except:
                st.info("Excel export requires openpyxl or xlsxwriter")
        
        st.success("✅ Performance data ready for download!")
        
    except Exception as e:
        st.error(f"Error exporting performance data: {str(e)}")


def export_strategic_insights():
    """Export strategic insights report."""
    
    try:
        # ✅ UPDATED: Use StateManager
        model_results = StateManager.get('model_results', {})
        best_model_name = StateManager.get('best_model_name')
        
        insights_content = "# CORTEXX STRATEGIC INSIGHTS REPORT\n\n"
        insights_content += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Best model insights
        if best_model_name:
            insights_content += f"## 🏆 BEST PERFORMING MODEL\n\n"
            insights_content += f"**{best_model_name}** is recommended for production deployment based on comprehensive evaluation.\n\n"
        
        # Performance insights with validation
        valid_rmse = []
        for results in model_results.values():
            rmse = results.get('test_rmse', results.get('rmse', float('inf')))
            if rmse != float('inf') and not np.isnan(rmse):
                valid_rmse.append(rmse)
        
        if valid_rmse:
            insights_content += "## 📈 PERFORMANCE INSIGHTS\n\n"
            insights_content += f"- **Performance Range**: RMSE varies from {min(valid_rmse):.4f} to {max(valid_rmse):.4f}\n"
            insights_content += f"- **Average Performance**: Mean RMSE is {np.mean(valid_rmse):.4f}\n"
            
            if len(valid_rmse) > 1:
                std_rmse = np.std(valid_rmse)
                variability = 'high' if std_rmse > np.mean(valid_rmse)*0.5 else 'moderate' if std_rmse > np.mean(valid_rmse)*0.2 else 'low'
                insights_content += f"- **Performance Spread**: Standard deviation of {std_rmse:.4f} indicates {variability} variability\n\n"
        
        # Strategic recommendations
        insights_content += "## 🎯 STRATEGIC RECOMMENDATIONS\n\n"
        insights_content += "### Immediate Actions (1-2 weeks)\n"
        insights_content += "- Deploy the best performing model to production environment\n"
        insights_content += "- Set up monitoring dashboards for performance tracking\n"
        insights_content += "- Establish baseline metrics for future comparisons\n\n"
        
        insights_content += "### Medium-term Initiatives (1-3 months)\n"
        insights_content += "- Implement automated retraining pipelines\n"
        insights_content += "- Develop ensemble modeling approaches\n"
        insights_content += "- Create A/B testing framework for model comparison\n\n"
        
        insights_content += "### Long-term Strategy (3-6 months)\n"
        insights_content += "- Build comprehensive model governance framework\n"
        insights_content += "- Implement MLOps practices for scalability\n"
        insights_content += "- Develop advanced feature engineering pipelines\n\n"
        
        # Risk assessment
        insights_content += "## ⚠️ RISK ASSESSMENT\n\n"
        insights_content += "- **Model Stability**: Monitor for performance degradation over time\n"
        insights_content += "- **Data Quality**: Ensure incoming data maintains expected patterns\n"
        insights_content += "- **Business Impact**: Consider confidence intervals in decision making\n"
        insights_content += "- **Regulatory Compliance**: Maintain model documentation and versioning\n"
        
        # Download insights
        st.markdown("### 💡 DOWNLOAD STRATEGIC INSIGHTS")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📋 INSIGHTS REPORT",
                data=insights_content,
                file_name=f"cortexx_strategic_insights_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        with col2:
            st.download_button(
                label="📄 EXECUTIVE SUMMARY",
                data=insights_content,
                file_name=f"cortexx_executive_summary_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        st.success("✅ Strategic insights ready for download!")
        
    except Exception as e:
        st.error(f"Error generating strategic insights: {str(e)}")


if __name__ == "__main__":
    main()
