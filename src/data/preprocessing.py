"""
Data preprocessing module for CortexX sales forecasting platform.
Handles data cleaning, missing value treatment, and feature encoding.

✅ FIXED ISSUES:
- Deprecated fillna(method=) replaced with ffill()/bfill()
- Added scaler persistence for production deployment
- Added inverse transform support
- Improved outlier handling with capping option
- Added fit/transform pattern for proper train/test splitting
- Enhanced error handling and logging
- Retail-specific validation
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from scipy import stats
import pickle
import os

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    A class to handle comprehensive data preprocessing for sales forecasting.
    
    ✅ ENHANCED FEATURES:
    - Scaler persistence for production deployment
    - Improved outlier handling (cap instead of remove)
    - Fit/transform pattern for proper train/test splitting
    - Inverse transformation support
    - Retail-specific validations
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.fitted_scalers = {}  # ✅ NEW: Store fitted scalers
        self.fitted_encoders = {}  # ✅ NEW: Store fitted encoders
        
    def handle_missing_values(
        self, 
        df: pd.DataFrame, 
        strategy: str = 'interpolate',
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset using specified strategy.
        
        ✅ FIXED: Deprecated pandas methods replaced
        
        Args:
            df (pd.DataFrame): Input dataframe
            strategy (str): Strategy for handling missing values
                - 'interpolate': Linear interpolation for numeric, forward fill for others
                - 'ffill': Forward fill
                - 'bfill': Backward fill
                - 'mean': Fill with column mean (numeric only)
                - 'median': Fill with column median (numeric only)
                - 'drop': Drop rows with missing values
            columns (List[str], optional): Specific columns to process
            
        Returns:
            pd.DataFrame: Dataframe with handled missing values
        """
        try:
            df_clean = df.copy()
            columns_to_process = columns if columns else df_clean.columns
            
            missing_before = df_clean[columns_to_process].isnull().sum().sum()
            
            for column in columns_to_process:
                if column not in df_clean.columns:
                    self.logger.warning(f"Column {column} not found in dataframe")
                    continue
                
                missing_count = df_clean[column].isnull().sum()
                if missing_count == 0:
                    continue
                    
                if strategy == 'interpolate':
                    if pd.api.types.is_numeric_dtype(df_clean[column]):
                        # ✅ FIXED: Use interpolate() with limit to avoid over-interpolation
                        df_clean[column] = df_clean[column].interpolate(
                            method='linear', 
                            limit_direction='both',
                            limit=5  # Limit interpolation to 5 consecutive missing values
                        )
                    else:
                        # ✅ FIXED: Use ffill() instead of fillna(method='ffill')
                        df_clean[column] = df_clean[column].ffill()
                        
                elif strategy == 'ffill':
                    # ✅ FIXED: Use ffill() instead of fillna(method='ffill')
                    df_clean[column] = df_clean[column].ffill()
                    
                elif strategy == 'bfill':
                    # ✅ FIXED: Use bfill() instead of fillna(method='bfill')
                    df_clean[column] = df_clean[column].bfill()
                    
                elif strategy == 'mean':
                    if pd.api.types.is_numeric_dtype(df_clean[column]):
                        mean_value = df_clean[column].mean()
                        df_clean[column] = df_clean[column].fillna(mean_value)
                    else:
                        self.logger.warning(f"Cannot apply mean to non-numeric column {column}")
                        
                elif strategy == 'median':
                    if pd.api.types.is_numeric_dtype(df_clean[column]):
                        median_value = df_clean[column].median()
                        df_clean[column] = df_clean[column].fillna(median_value)
                    else:
                        self.logger.warning(f"Cannot apply median to non-numeric column {column}")
                        
                elif strategy == 'drop':
                    df_clean = df_clean.dropna(subset=[column])
                    
                else:
                    self.logger.warning(f"Unknown strategy: {strategy}. Using forward fill.")
                    df_clean[column] = df_clean[column].ffill()
            
            missing_after = df_clean[columns_to_process].isnull().sum().sum()
            self.logger.info(
                f"✅ Handled {missing_before - missing_after} missing values using {strategy} strategy. "
                f"Remaining: {missing_after}"
            )
            
            return df_clean
            
        except Exception as e:
            self.logger.error(f"Error handling missing values: {str(e)}")
            raise ValueError(f"Missing value handling failed: {str(e)}")
    
    def handle_outliers(
        self, 
        df: pd.DataFrame, 
        column: str, 
        method: str = 'iqr',
        action: str = 'cap',  # ✅ CHANGED: Default to 'cap' instead of 'remove'
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Handle outliers in specified column using various methods.
        
        ✅ IMPROVED: Changed default action from 'remove' to 'cap' to preserve data
        
        Args:
            df (pd.DataFrame): Input dataframe
            column (str): Column name to process
            method (str): Method for outlier detection ('iqr', 'zscore', 'percentile')
            action (str): Action to take ('cap', 'remove', 'flag')
            threshold (float): Threshold for outlier detection
            
        Returns:
            pd.DataFrame: Dataframe with handled outliers
        """
        try:
            if column not in df.columns:
                self.logger.warning(f"Column {column} not found in dataframe")
                return df.copy()
            
            df_clean = df.copy()
            
            # Calculate bounds based on method
            if method == 'iqr':
                Q1 = df_clean[column].quantile(0.25)
                Q3 = df_clean[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
            elif method == 'zscore':
                mean = df_clean[column].mean()
                std = df_clean[column].std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
                
            elif method == 'percentile':
                lower_bound = df_clean[column].quantile(0.01)
                upper_bound = df_clean[column].quantile(0.99)
                
            else:
                self.logger.warning(f"Unknown outlier method: {method}")
                return df_clean
            
            # Identify outliers
            outlier_mask = (df_clean[column] < lower_bound) | (df_clean[column] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            # Take action
            if action == 'cap':
                # ✅ NEW: Cap outliers instead of removing (preserves data)
                df_clean.loc[df_clean[column] < lower_bound, column] = lower_bound
                df_clean.loc[df_clean[column] > upper_bound, column] = upper_bound
                self.logger.info(
                    f"✅ Capped {outlier_count} outliers in {column} to "
                    f"[{lower_bound:.2f}, {upper_bound:.2f}]"
                )
                
            elif action == 'remove':
                initial_count = len(df_clean)
                df_clean = df_clean[~outlier_mask]
                removed_count = initial_count - len(df_clean)
                self.logger.info(f"⚠️ Removed {removed_count} outlier rows from {column}")
                
            elif action == 'flag':
                # ✅ NEW: Flag outliers without modifying data
                df_clean[f'{column}_outlier'] = outlier_mask.astype(int)
                self.logger.info(f"✅ Flagged {outlier_count} outliers in {column}")
                
            else:
                self.logger.warning(f"Unknown action: {action}. No changes made.")
            
            return df_clean
            
        except Exception as e:
            self.logger.error(f"Error handling outliers: {str(e)}")
            raise ValueError(f"Outlier handling failed: {str(e)}")
    
    def fit_encoder(
        self, 
        df: pd.DataFrame, 
        columns: Optional[List[str]] = None,
        method: str = 'onehot'
    ) -> 'DataPreprocessor':
        """
        ✅ NEW: Fit encoder on training data (for proper train/test split).
        
        Args:
            df (pd.DataFrame): Training dataframe
            columns (List[str], optional): Categorical columns to encode
            method (str): Encoding method ('onehot', 'label')
        
        Returns:
            self: For method chaining
        """
        try:
            if columns is None:
                # Auto-detect categorical columns
                categorical_columns = df.select_dtypes(
                    include=['object', 'category']
                ).columns.tolist()
            else:
                categorical_columns = [col for col in columns if col in df.columns]
            
            if not categorical_columns:
                self.logger.info("No categorical columns found for encoding")
                return self
            
            if method == 'label':
                for col in categorical_columns:
                    le = LabelEncoder()
                    le.fit(df[col].astype(str))
                    self.fitted_encoders[col] = le
                    
            elif method == 'onehot':
                # Store column names for one-hot encoding
                self.fitted_encoders['onehot_columns'] = categorical_columns
                # Get unique values for each column
                for col in categorical_columns:
                    self.fitted_encoders[f'onehot_{col}_categories'] = df[col].unique()
            
            self.logger.info(f"✅ Fitted encoder for {len(categorical_columns)} columns using {method}")
            return self
            
        except Exception as e:
            self.logger.error(f"Error fitting encoder: {str(e)}")
            raise ValueError(f"Encoder fitting failed: {str(e)}")
    
    def transform_categorical(
        self, 
        df: pd.DataFrame,
        method: str = 'onehot'
    ) -> pd.DataFrame:
        """
        ✅ NEW: Transform categorical variables using fitted encoder.
        
        Args:
            df (pd.DataFrame): Dataframe to transform
            method (str): Encoding method (must match fit method)
        
        Returns:
            pd.DataFrame: Dataframe with encoded categorical variables
        """
        try:
            df_encoded = df.copy()
            
            if method == 'label':
                for col, encoder in self.fitted_encoders.items():
                    if col in df_encoded.columns and not col.startswith('onehot_'):
                        # Handle unseen categories
                        df_encoded[col] = df_encoded[col].astype(str).map(
                            lambda x: encoder.transform([x])[0] 
                            if x in encoder.classes_ 
                            else -1
                        )
                        
            elif method == 'onehot':
                categorical_columns = self.fitted_encoders.get('onehot_columns', [])
                if categorical_columns:
                    df_encoded = pd.get_dummies(
                        df_encoded, 
                        columns=categorical_columns, 
                        prefix=categorical_columns
                    )
            
            self.logger.info(f"✅ Transformed categorical variables using {method} encoding")
            return df_encoded
            
        except Exception as e:
            self.logger.error(f"Error transforming categorical variables: {str(e)}")
            raise ValueError(f"Categorical transformation failed: {str(e)}")
    
    def encode_categorical_variables(
        self, 
        df: pd.DataFrame, 
        columns: Optional[List[str]] = None,
        method: str = 'onehot'
    ) -> pd.DataFrame:
        """
        Encode categorical variables (fit and transform in one step).
        
        ⚠️ WARNING: Use fit_encoder() + transform_categorical() for train/test splits.
        This method is for convenience when you don't need separate fit/transform.
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (List[str], optional): Categorical columns to encode
            method (str): Encoding method ('onehot', 'label')
        
        Returns:
            pd.DataFrame: Dataframe with encoded categorical variables
        """
        self.fit_encoder(df, columns, method)
        return self.transform_categorical(df, method)
    
    def fit_scaler(
        self, 
        df: pd.DataFrame, 
        columns: Optional[List[str]] = None,
        method: str = 'standardize',
        scaler_name: str = 'default'
    ) -> 'DataPreprocessor':
        """
        ✅ NEW: Fit scaler on training data.
        
        Args:
            df (pd.DataFrame): Training dataframe
            columns (List[str], optional): Numerical columns to normalize
            method (str): Normalization method ('standardize', 'minmax', 'robust')
            scaler_name (str): Name to store scaler under (for multiple scalers)
        
        Returns:
            self: For method chaining
        """
        try:
            if columns is None:
                numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            else:
                numerical_columns = [
                    col for col in columns 
                    if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
                ]
            
            if not numerical_columns:
                self.logger.info("No numerical columns found for normalization")
                return self
            
            # Select scaler
            if method == 'standardize':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                self.logger.warning(f"Unknown normalization method: {method}. Using standard scaling.")
                scaler = StandardScaler()
            
            # Fit scaler
            scaler.fit(df[numerical_columns])
            
            # Store scaler and columns
            self.fitted_scalers[scaler_name] = {
                'scaler': scaler,
                'columns': numerical_columns,
                'method': method
            }
            
            self.logger.info(
                f"✅ Fitted {method} scaler for {len(numerical_columns)} columns "
                f"(stored as '{scaler_name}')"
            )
            return self
            
        except Exception as e:
            self.logger.error(f"Error fitting scaler: {str(e)}")
            raise ValueError(f"Scaler fitting failed: {str(e)}")
    
    def transform_numerical(
        self, 
        df: pd.DataFrame,
        scaler_name: str = 'default'
    ) -> pd.DataFrame:
        """
        ✅ NEW: Transform numerical data using fitted scaler.
        
        Args:
            df (pd.DataFrame): Dataframe to transform
            scaler_name (str): Name of fitted scaler to use
        
        Returns:
            pd.DataFrame: Dataframe with normalized numerical data
        """
        try:
            if scaler_name not in self.fitted_scalers:
                raise ValueError(f"Scaler '{scaler_name}' not fitted. Call fit_scaler() first.")
            
            df_normalized = df.copy()
            scaler_info = self.fitted_scalers[scaler_name]
            scaler = scaler_info['scaler']
            columns = scaler_info['columns']
            
            # Transform
            df_normalized[columns] = scaler.transform(df_normalized[columns])
            
            self.logger.info(f"✅ Transformed {len(columns)} numerical columns using '{scaler_name}' scaler")
            return df_normalized
            
        except Exception as e:
            self.logger.error(f"Error transforming numerical data: {str(e)}")
            raise ValueError(f"Numerical transformation failed: {str(e)}")
    
    def inverse_transform_numerical(
        self, 
        df: pd.DataFrame,
        scaler_name: str = 'default'
    ) -> pd.DataFrame:
        """
        ✅ NEW: Inverse transform numerical data back to original scale.
        
        Args:
            df (pd.DataFrame): Normalized dataframe
            scaler_name (str): Name of fitted scaler to use
        
        Returns:
            pd.DataFrame: Dataframe with denormalized numerical data
        """
        try:
            if scaler_name not in self.fitted_scalers:
                raise ValueError(f"Scaler '{scaler_name}' not fitted.")
            
            df_denormalized = df.copy()
            scaler_info = self.fitted_scalers[scaler_name]
            scaler = scaler_info['scaler']
            columns = scaler_info['columns']
            
            # Inverse transform
            df_denormalized[columns] = scaler.inverse_transform(df_denormalized[columns])
            
            self.logger.info(f"✅ Inverse transformed {len(columns)} numerical columns")
            return df_denormalized
            
        except Exception as e:
            self.logger.error(f"Error inverse transforming data: {str(e)}")
            raise ValueError(f"Inverse transformation failed: {str(e)}")
    
    def normalize_data(
        self, 
        df: pd.DataFrame, 
        columns: Optional[List[str]] = None,
        method: str = 'standardize'
    ) -> Tuple[pd.DataFrame, Any]:
        """
        Normalize numerical data (fit and transform in one step).
        
        ⚠️ WARNING: Use fit_scaler() + transform_numerical() for train/test splits.
        
        Returns:
            Tuple[pd.DataFrame, scaler]: Normalized dataframe and fitted scaler
        """
        self.fit_scaler(df, columns, method, scaler_name='temp')
        df_normalized = self.transform_numerical(df, scaler_name='temp')
        scaler = self.fitted_scalers['temp']['scaler']
        return df_normalized, scaler
    
    def save_preprocessor(self, filepath: str):
        """
        ✅ NEW: Save fitted scalers and encoders for production deployment.
        
        Args:
            filepath (str): Path to save preprocessor state
        """
        try:
            state = {
                'fitted_scalers': self.fitted_scalers,
                'fitted_encoders': self.fitted_encoders
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            self.logger.info(f"✅ Saved preprocessor state to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving preprocessor: {str(e)}")
            raise
    
    def load_preprocessor(self, filepath: str):
        """
        ✅ NEW: Load fitted scalers and encoders.
        
        Args:
            filepath (str): Path to load preprocessor state from
        """
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            self.fitted_scalers = state['fitted_scalers']
            self.fitted_encoders = state['fitted_encoders']
            self.logger.info(f"✅ Loaded preprocessor state from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading preprocessor: {str(e)}")
            raise
    
    def validate_retail_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ✅ NEW: Validate and clean retail-specific data issues.
        
        Args:
            df (pd.DataFrame): Input dataframe
        
        Returns:
            pd.DataFrame: Validated dataframe
        """
        df_clean = df.copy()
        issues_fixed = []
        
        try:
            # Check for negative sales
            if 'Units Sold' in df_clean.columns:
                negative_sales = (df_clean['Units Sold'] < 0).sum()
                if negative_sales > 0:
                    self.logger.warning(f"⚠️ Found {negative_sales} negative sales. Setting to 0.")
                    df_clean.loc[df_clean['Units Sold'] < 0, 'Units Sold'] = 0
                    issues_fixed.append(f"Fixed {negative_sales} negative sales")
            
            # Check for unrealistic sales (> 99th percentile by 10x)
            if 'Units Sold' in df_clean.columns:
                p99 = df_clean['Units Sold'].quantile(0.99)
                extreme_mask = df_clean['Units Sold'] > (p99 * 10)
                extreme_count = extreme_mask.sum()
                if extreme_count > 0:
                    self.logger.warning(f"⚠️ Found {extreme_count} extreme sales values. Capping.")
                    df_clean.loc[extreme_mask, 'Units Sold'] = p99 * 10
                    issues_fixed.append(f"Capped {extreme_count} extreme sales")
            
            # Check for negative prices
            price_cols = ['Price', 'Competitor Pricing']
            for col in price_cols:
                if col in df_clean.columns:
                    negative_prices = (df_clean[col] < 0).sum()
                    if negative_prices > 0:
                        self.logger.warning(f"⚠️ Found {negative_prices} negative {col}. Taking absolute value.")
                        df_clean[col] = df_clean[col].abs()
                        issues_fixed.append(f"Fixed {negative_prices} negative {col}")
            
            if issues_fixed:
                self.logger.info(f"✅ Retail validation complete: {', '.join(issues_fixed)}")
            else:
                self.logger.info("✅ No retail validation issues found")
            
            return df_clean
            
        except Exception as e:
            self.logger.error(f"Error in retail validation: {str(e)}")
            return df_clean


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_preprocessor() -> DataPreprocessor:
    """Get or create DataPreprocessor instance."""
    return DataPreprocessor()
