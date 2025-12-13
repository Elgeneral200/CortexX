"""
✅ PHASE 2+3 UI - FULLY INTEGRATED with Static FeatureEngineer + State Management
- Uses static methods (no instantiation)
- Caches engineered data in state
- Shows performance timing
- Prevents re-engineering
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
import os
import time
from datetime import datetime

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

try:
    from src.utils.state_manager import StateManager, is_data_loaded, get_current_data
    from src.features.engineering import FeatureEngineer
    from src.features.selection import FeatureSelector
    from src.visualization.dashboard import get_visualizer, display_plotly_chart
    MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"Feature engineering modules not available: {e}")
    MODULES_AVAILABLE = False

st.set_page_config(page_title="Feature Engineering - CortexX", page_icon="⚙️", layout="wide")


def safe_clean_df(df):
    """🚨 BULLETPROOF: Clean ANY dataframe."""
    if df is None:
        return pd.DataFrame()
    
    # Method 1: Remove duplicates
    df1 = df.loc[:, ~df.columns.duplicated(keep='first')]
    
    # Method 2: Ensure unique columns
    unique_cols = []
    for col in df1.columns:
        if col not in unique_cols:
            unique_cols.append(col)
    
    df_clean = df1[unique_cols].copy()
    return df_clean


def safe_display_df(df, cols, title="Data Preview"):
    """🚨 BULLETPROOF: Display ANY dataframe safely."""
    try:
        df_clean = safe_clean_df(df)
        safe_cols = [c for c in cols if c in df_clean.columns]
        
        if safe_cols:
            display_df = df_clean[safe_cols].head(10)
        else:
            display_df = df_clean.head(10)
        
        st.dataframe(display_df, use_container_width=True)
        return True
    except:
        st.info("📊 Dataset ready (display limited)")
        return False


def main():
    st.markdown('<div class="section-header">⚙️ ENTERPRISE FEATURE ENGINEERING + SELECTION</div>', unsafe_allow_html=True)
    
    StateManager.initialize()
    
    # Check if data is loaded
    if not is_data_loaded():
        st.warning("⚠️ Please load data first from the Dashboard page")
        return
    
    if not MODULES_AVAILABLE:
        st.error("❌ Feature engineering modules not available.")
        return
    
    # NEW: Check if already engineered (FAST PATH)
    if StateManager.is_data_engineered():
        render_cached_features()
        return
    
    # SLOW PATH: Need to engineer features
    df = get_current_data().copy()
    df_clean = safe_clean_df(df)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚀 Quick Retail Pipeline", "📅 Time Features", "🔄 Lag/Rolling", "🎯 Selection", "📊 Summary"
    ])
    
    with tab1: render_quick_pipeline(df_clean)
    with tab2: render_time_features(df_clean)
    with tab3: render_lag_rolling(df_clean)
    with tab4: render_selection(df_clean)
    with tab5: render_summary(df_clean)


def render_cached_features():
    """Display already engineered features (FAST!)"""
    st.info("✅ Features already engineered! Using cached version.")
    
    df_engineered = StateManager.get_engineered_data()
    eng_time = StateManager.get('feature_engineering_time')
    selected_features = StateManager.get('selected_features', [])
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Rows", f"{len(df_engineered):,}")
    with col2:
        st.metric("📈 Total Features", len(df_engineered.columns))
    with col3:
        original_cols = len(get_current_data().columns)
        new_features = len(df_engineered.columns) - original_cols
        st.metric("✨ New Features", new_features)
    with col4:
        st.metric("🎯 Selected", len(selected_features))
    
    if eng_time:
        st.caption(f"⏰ Created: {eng_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Preview
    with st.expander("📊 View Engineered Data", expanded=False):
        preview_cols = df_engineered.columns[:15].tolist()
        safe_display_df(df_engineered, preview_cols, "Engineered Features Preview")
    
    # Feature breakdown
    with st.expander("📋 Feature Categories", expanded=False):
        feature_types = {
            'Time Features': [c for c in df_engineered.columns if any(x in c.lower() for x in ['year', 'month', 'day', 'week', 'quarter'])],
            'Lag Features': [c for c in df_engineered.columns if 'lag' in c.lower()],
            'Rolling Features': [c for c in df_engineered.columns if 'roll' in c.lower()],
            'Price Features': [c for c in df_engineered.columns if 'price' in c.lower()],
            'Promotion Features': [c for c in df_engineered.columns if 'promo' in c.lower()],
            'Inventory Features': [c for c in df_engineered.columns if 'inventory' in c.lower() or 'stock' in c.lower()],
            'Hierarchical Features': [c for c in df_engineered.columns if any(x in c.lower() for x in ['store', 'product', 'category'])],
        }
        
        for category, features in feature_types.items():
            if features:
                st.markdown(f"**{category}** ({len(features)} features)")
                st.caption(", ".join(features[:10]) + ("..." if len(features) > 10 else ""))
    
    # Selected features
    if selected_features:
        with st.expander("🎯 Selected Features for Training", expanded=True):
            st.success(f"**{len(selected_features)} features selected for model training**")
            for i, feat in enumerate(selected_features[:20], 1):
                st.code(f"{i}. {feat}")
            if len(selected_features) > 20:
                st.caption(f"... and {len(selected_features) - 20} more")
    
    # Actions
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Re-engineer Features", type="secondary", use_container_width=True):
            StateManager.clear_engineered_data()
            st.rerun()
    
    with col2:
        if st.button("➡️ Go to Model Training", type="primary", use_container_width=True):
            st.switch_page("pages/4__Model_Training.py")
    
    st.info("✨ Your engineered features are ready for Model Training!")


def render_quick_pipeline(df):
    """Full retail pipeline with timing and state management"""
    st.markdown("### 🚀 Complete Retail Feature Pipeline")
    st.markdown("Creates **all retail-optimized features** in one click:")
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown("**Pipeline Steps:**")
        st.markdown("""
        1. ⏰ Time features (year, month, day, cyclical)
        2. 📊 Lag features (7, 14, 30 days)
        3. 📈 Rolling statistics (7, 14, 30 day windows)
        4. 💰 Price & discount features
        5. 🎉 Promotion features
        6. 📦 Inventory features
        7. 🏪 Store & product hierarchies
        8. 🎯 Automatic feature selection
        """)
        
        st.markdown("---")
        
        # Configuration
        n_features = st.slider("🎯 Select Top N Features", 10, 100, 30, 
                               help="Number of features to select after engineering")
        
        # Run button
        if st.button("🚀 RUN FULL PIPELINE", type="primary", use_container_width=True):
            run_full_pipeline(df, n_features)
    
    with col2:
        # Preview current state
        selected = StateManager.get('selected_features', [])
        if selected:
            st.success(f"**✅ Pipeline Complete!**")
            st.metric("🎯 Selected Features", len(selected))
            
            with st.expander("📋 Top Features", expanded=True):
                for i, feat in enumerate(selected[:15], 1):
                    st.code(f"{i}. {feat}", language="")
                if len(selected) > 15:
                    st.caption(f"... and {len(selected) - 15} more")
        else:
            st.info("👆 Click **RUN FULL PIPELINE** to start")


def run_full_pipeline(df, n_features=30):
    """Execute full feature engineering pipeline with timing"""
    
    progress = st.progress(0, text="Initializing pipeline...")
    timing_info = {}
    
    try:
        # Validate columns
        required_cols = {
            'store_col': 'Store ID',
            'product_col': 'Product ID',
            'value_col': 'Units Sold',
            'date_col': StateManager.get('date_column', 'Date')
        }
        
        for key, col in required_cols.items():
            if col not in df.columns:
                st.error(f"❌ Required column '{col}' not found in data!")
                return
        
        progress.progress(10, text="🏭 Creating retail features...")
        overall_start = time.time()
        
        # Call static method (NO INSTANTIATION!)
        start = time.time()
        df_engineered = FeatureEngineer.create_retail_features(
            df,
            store_col=required_cols['store_col'],
            product_col=required_cols['product_col'],
            value_col=required_cols['value_col'],
            date_col=required_cols['date_col']
        )
        timing_info['feature_engineering'] = time.time() - start
        
        progress.progress(70, text="🎯 Selecting best features...")
        
        # Feature selection (also uses instance, but that's OK for now)
        start = time.time()
        selector = FeatureSelector()
        selected = selector.select_hierarchical_features(
            df_engineered, 
            required_cols['value_col'], 
            n_features=n_features
        )
        timing_info['feature_selection'] = time.time() - start
        
        timing_info['total'] = time.time() - overall_start
        
        progress.progress(90, text="💾 Saving to state...")
        
        # Calculate new features
        original_cols = list(df.columns)
        feature_list = [col for col in df_engineered.columns if col not in original_cols]
        
        # SAVE TO STATE (CRITICAL!)
        StateManager.set_engineered_data(df_engineered, feature_list)
        StateManager.set('selected_features', selected)
        
        progress.progress(100, text="✅ Complete!")
        
        # Success message with timing
        st.success(f"✅ **Pipeline Complete in {timing_info['total']:.1f}s!**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("⚙️ Engineering", f"{timing_info['feature_engineering']:.1f}s")
        with col2:
            st.metric("🎯 Selection", f"{timing_info['feature_selection']:.1f}s")
        with col3:
            st.metric("✨ Features Created", len(feature_list))
        
        st.info(f"📊 Selected **{len(selected)}** optimal features for training")
        st.balloons()
        
        time.sleep(1)
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Pipeline failed: {str(e)}")
        st.exception(e)
    finally:
        progress.empty()


def render_time_features(df):
    """Individual time feature creation"""
    st.markdown("### 📅 Time-Based Features")
    
    date_col = StateManager.get('date_column')
    if not date_col or date_col not in df.columns:
        st.warning("⚠️ No date column detected. Please configure in Dashboard.")
        return
    
    st.markdown(f"**Using date column:** `{date_col}`")
    
    st.markdown("""
    **Will create:**
    - Year, Month, Quarter, Week, Day
    - Day of week, Day of month, Day of year
    - Binary flags (weekend, month start/end, quarter start/end)
    - Cyclical encodings (sin/cos transformations)
    """)
    
    if st.button("🛠️ CREATE TIME FEATURES", type="primary"):
        with st.spinner("Creating time features..."):
            start = time.time()
            
            # Use static method
            df_time = FeatureEngineer.create_time_features(df, date_col)
            
            elapsed = time.time() - start
            
            # Update state
            StateManager.set('current_data', df_time)
            
            new_features = len(df_time.columns) - len(df.columns)
            st.success(f"✅ Created {new_features} time features in {elapsed:.2f}s!")
            st.rerun()


def render_lag_rolling(df):
    """Individual lag and rolling feature creation"""
    st.markdown("### 🔄 Lag & Rolling Features")
    
    value_col = StateManager.get('value_column', 'Units Sold')
    if value_col not in df.columns:
        st.warning(f"⚠️ Value column '{value_col}' not found.")
        return
    
    st.markdown(f"**Using value column:** `{value_col}`")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Lag Features:**")
        lags = st.multiselect("Lag periods (days)", [1, 7, 14, 21, 30, 60], default=[7, 14, 30])
    
    with col2:
        st.markdown("**Rolling Windows:**")
        windows = st.multiselect("Window sizes (days)", [7, 14, 21, 30, 60, 90], default=[7, 14, 30])
    
    group_cols = ['Store ID', 'Product ID']
    st.info(f"📊 Will create features grouped by: {', '.join(group_cols)}")
    
    if st.button("🔄 CREATE LAG/ROLLING FEATURES", type="primary"):
        with st.spinner("Creating lag and rolling features..."):
            start = time.time()
            
            # Use static methods
            df_result = df.copy()
            
            if lags:
                df_result = FeatureEngineer.create_lag_features(
                    df_result, value_col, lags, group_cols
                )
            
            if windows:
                df_result = FeatureEngineer.create_rolling_features(
                    df_result, value_col, windows, group_cols
                )
            
            elapsed = time.time() - start
            
            # Update state
            StateManager.set('current_data', df_result)
            
            new_features = len(df_result.columns) - len(df.columns)
            st.success(f"✅ Created {new_features} features in {elapsed:.2f}s!")
            st.rerun()


def render_selection(df):
    """Feature selection interface"""
    st.markdown("### 🎯 Feature Selection")
    
    value_col = StateManager.get('value_column', 'Units Sold')
    if value_col not in df.columns:
        st.warning(f"⚠️ Target column '{value_col}' not found.")
        return
    
    st.markdown(f"**Using target column:** `{value_col}`")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Selection Method:**")
        method = st.radio(
            "Choose method",
            ["Random Forest", "F-Regression", "Mutual Info"],
            help="Random Forest: Best for grouped data (stores/products)"
        )
        
        method_map = {
            "Random Forest": "randomforest",
            "F-Regression": "fregression",
            "Mutual Info": "mutualinfo"
        }
        
        n_features = st.slider("Number of features", 10, 100, 30)
        
        if st.button("🎯 SELECT FEATURES", type="primary"):
            with st.spinner("Selecting features..."):
                start = time.time()
                
                selector = FeatureSelector()
                group_cols = ['Store ID', 'Product ID']
                
                try:
                    selected = selector.select_hierarchical_features(
                        df, 
                        value_col, 
                        group_cols=group_cols,
                        n_features=n_features,
                        method=method_map[method]
                    )
                    
                    elapsed = time.time() - start
                    StateManager.set('selected_features', selected)
                    st.success(f"✅ Selected {len(selected)} features in {elapsed:.2f}s!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Selection failed: {str(e)}")
                    st.exception(e)
    
    with col2:
        selected = StateManager.get('selected_features', [])
        if selected:
            st.metric("✅ Selected Features", len(selected))
            
            try:
                selector = FeatureSelector()
                imp_df = selector.calculate_grouped_importance(
                    df, 
                    value_col,
                    group_cols=['Store ID', 'Product ID'],
                    method='randomforest'
                )
                
                if not imp_df.empty:
                    fig = px.bar(
                        imp_df.head(15), 
                        x='importance', 
                        y='feature', 
                        orientation='h',
                        title="Top 15 Features by Importance"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except:
                st.info("Feature importance calculation not available")
            
            with st.expander("📋 Selected Features List"):
                for i, feat in enumerate(selected, 1):
                    st.code(f"{i}. {feat}", language="")
        else:
            st.info("👆 Select features to see them here")



def render_summary(df):
    """Dataset and feature summary"""
    st.markdown("### 📊 Dataset Summary")
    
    df_clean = safe_clean_df(df)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Columns", len(df_clean.columns))
    with col2:
        numeric_cols = df_clean.select_dtypes(include=np.number).columns
        st.metric("Numeric", len(numeric_cols))
    with col3:
        selected = StateManager.get('selected_features', [])
        st.metric("Selected", len(selected))
    with col4:
        st.metric("Rows", f"{len(df_clean):,}")
    
    # Data preview
    st.markdown("**Data Preview:**")
    preview_cols = df_clean.columns[:15].tolist()
    safe_display_df(df_clean, preview_cols, "Current Dataset")
    
    # Column types
    with st.expander("📋 Column Types"):
        col_types = df_clean.dtypes.value_counts()
        for dtype, count in col_types.items():
            st.markdown(f"- **{dtype}**: {count} columns")


if __name__ == "__main__":
    main()
