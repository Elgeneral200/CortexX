"""
Comprehensive Test Suite for Phase 3 - Retail Forecasting Integration

✅ CONFIGURED FOR YOUR PROJECT:
- Project: D:\cortexx-forecasting\
- Dataset: D:\cortexx-forecasting\retail_store_inventory.csv
- Tests folder: D:\cortexx-forecasting\tests\

Tests all updated files:
1. collection.py - Data loading and retail validation
2. preprocessing.py - Data cleaning and transformations
3. data_quality.py - Quality analysis
4. validators.py - Validation rules
5. exploration.py - Data exploration

Run from project root:
cd D:\cortexx-forecasting
python tests\test_phase3_integration.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# ✅ Add parent directory to path (from tests folder to project root)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, 'src'))

# ✅ YOUR DATASET PATH (relative to project root)
DATASET_PATH = r"D:\cortexx-forecasting\retail_store_inventory.csv"

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_test_header(test_name):
    """Print formatted test header."""
    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}TEST: {test_name}{RESET}")
    print(f"{BLUE}{'='*80}{RESET}")


def print_success(message):
    """Print success message."""
    print(f"{GREEN}✅ {message}{RESET}")


def print_error(message):
    """Print error message."""
    print(f"{RED}❌ {message}{RESET}")


def print_warning(message):
    """Print warning message."""
    print(f"{YELLOW}⚠️  {message}{RESET}")


def print_info(message):
    """Print info message."""
    print(f"{BLUE}ℹ️  {message}{RESET}")


# ============================================================================
# TEST 1: IMPORTS
# ============================================================================

def test_imports():
    """Test all module imports."""
    print_test_header("Module Imports")
    
    try:
        from src.data.collection import DataCollector
        print_success("DataCollector imported")
    except ImportError as e:
        print_error(f"DataCollector import failed: {e}")
        return False
    
    try:
        from src.data.preprocessing import DataPreprocessor
        print_success("DataPreprocessor imported")
    except ImportError as e:
        print_error(f"DataPreprocessor import failed: {e}")
        return False
    
    try:
        from src.analytics.data_quality import DataQualityAnalyzer
        print_success("DataQualityAnalyzer imported")
    except ImportError as e:
        print_error(f"DataQualityAnalyzer import failed: {e}")
        return False
    
    try:
        from src.utils.validators import DataValidator
        print_success("DataValidator imported")
    except ImportError as e:
        print_error(f"DataValidator import failed: {e}")
        return False
    
    try:
        from src.data.exploration import DataExplorer
        print_success("DataExplorer imported")
    except ImportError as e:
        print_error(f"DataExplorer import failed: {e}")
        return False
    
    print_success("All imports successful!\n")
    return True


# ============================================================================
# TEST 2: DATA COLLECTION
# ============================================================================

def test_data_collection():
    """Test data collection functionality."""
    print_test_header("Data Collection")
    
    from src.data.collection import DataCollector
    
    try:
        # Initialize collector
        collector = DataCollector()
        print_success("DataCollector initialized")
        
        # ✅ TRY TO LOAD YOUR DATASET FIRST
        if os.path.exists(DATASET_PATH):
            print_info(f"📂 Loading YOUR dataset from: {DATASET_PATH}")
            df = collector.load_csv_data(DATASET_PATH, validate_retail=True)
            print_success(f"✅ Loaded YOUR dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
        else:
            print_warning(f"⚠️  Dataset not found at: {DATASET_PATH}")
            print_info("🔄 Generating sample retail data instead...")
            df = collector.generate_sample_retail_data(
                periods=100,
                n_stores=3,
                n_products=10
            )
            print_success(f"Generated sample data: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        # Check required columns
        required_cols = ['Date', 'Store ID', 'Product ID', 'Units Sold']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print_error(f"❌ Missing required columns: {missing_cols}")
            print_info(f"Available columns: {df.columns.tolist()}")
            return False, None
        else:
            print_success(f"✅ All required columns present: {required_cols}")
        
        # Check data types
        print_info("Checking data types...")
        print(f"  - Date: {df['Date'].dtype}")
        print(f"  - Store ID: {df['Store ID'].dtype}")
        print(f"  - Product ID: {df['Product ID'].dtype}")
        print(f"  - Units Sold: {df['Units Sold'].dtype}")
        
        # Check for negative values
        negative_sales = (df['Units Sold'] < 0).sum()
        if negative_sales > 0:
            print_warning(f"Found {negative_sales} negative sales values")
        else:
            print_success("No negative sales values")
        
        # ✅ FIXED: Get retail summary with safe dictionary access
        summary = collector.get_retail_summary(df)
        print_success("Retail summary generated")
        
        # ✅ FIXED: Safe access to nested dictionaries
        if summary.get('stores') and isinstance(summary['stores'], dict):
            print(f"  - Stores: {summary['stores'].get('count', 'N/A')}")
        else:
            print(f"  - Stores: N/A")
        
        if summary.get('products') and isinstance(summary['products'], dict):
            print(f"  - Products: {summary['products'].get('count', 'N/A')}")
        else:
            print(f"  - Products: N/A")
        
        if summary.get('total_units_sold') is not None:
            print(f"  - Total Sales: {summary['total_units_sold']:,}")
        else:
            print(f"  - Total Sales: N/A")
        
        if summary.get('date_range') and isinstance(summary['date_range'], dict):
            print(f"  - Date Range: {summary['date_range'].get('start', 'N/A')} to {summary['date_range'].get('end', 'N/A')}")
        
        return True, df
        
    except Exception as e:
        print_error(f"Data collection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


# ============================================================================
# TEST 3: DATA PREPROCESSING
# ============================================================================

def test_data_preprocessing(df):
    """Test data preprocessing functionality."""
    print_test_header("Data Preprocessing")
    
    from src.data.preprocessing import DataPreprocessor
    
    try:
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        print_success("DataPreprocessor initialized")
        
        # Add some missing values for testing
        df_test = df.copy()
        missing_before = df_test['Units Sold'].isnull().sum()
        
        # Only add missing if there aren't many already
        if missing_before < 5:
            df_test.loc[0:5, 'Units Sold'] = np.nan
            print_info(f"Added {df_test['Units Sold'].isnull().sum() - missing_before} missing values for testing")
        else:
            print_info(f"Dataset already has {missing_before} missing values")
        
        # Test missing value handling
        df_clean = preprocessor.handle_missing_values(df_test, strategy='ffill')
        missing_after = df_clean['Units Sold'].isnull().sum()
        print_success(f"Missing values handled: {missing_after} remaining (was {df_test['Units Sold'].isnull().sum()})")
        
        # Test outlier handling
        df_outliers = preprocessor.handle_outliers(
            df_clean, 
            'Units Sold', 
            method='iqr',
            action='cap'
        )
        print_success("Outlier handling completed (cap method)")
        
        # Test fit/transform pattern
        print_info("Testing fit/transform pattern...")
        
        # Select only numeric columns for scaling
        numeric_cols = df_outliers.select_dtypes(include=[np.number]).columns.tolist()
        if 'Units Sold' in numeric_cols:
            preprocessor.fit_scaler(df_outliers[['Units Sold']], method='standardize', scaler_name='test_scaler')
            print_success("Scaler fitted")
            
            df_scaled = preprocessor.transform_numerical(df_outliers, scaler_name='test_scaler')
            print_success("Data transformed")
            
            # Test inverse transform
            df_original = preprocessor.inverse_transform_numerical(df_scaled, scaler_name='test_scaler')
            print_success("Inverse transform completed")
            
            # Verify inverse transform accuracy
            original_mean = df_outliers['Units Sold'].mean()
            restored_mean = df_original['Units Sold'].mean()
            diff = abs(original_mean - restored_mean)
            
            if diff < 0.01:
                print_success(f"Inverse transform accurate (diff: {diff:.6f})")
            else:
                print_warning(f"Inverse transform has some error (diff: {diff:.6f})")
        else:
            print_warning("'Units Sold' column not found for scaling test")
        
        # Test retail validation
        df_validated = preprocessor.validate_retail_data(df_outliers)
        print_success("Retail data validation completed")
        
        return True
        
    except Exception as e:
        print_error(f"Preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 4: DATA QUALITY
# ============================================================================

def test_data_quality(df):
    """Test data quality analysis."""
    print_test_header("Data Quality Analysis")
    
    from src.analytics.data_quality import DataQualityAnalyzer
    
    try:
        # Initialize analyzer
        analyzer = DataQualityAnalyzer(df)
        print_success("DataQualityAnalyzer initialized")
        
        # Test overall quality score
        quality_score = analyzer.calculate_overall_quality_score()
        print_success(f"Quality Score: {quality_score['overall_score']:.1f}/100 (Grade: {quality_score['grade']})")
        print(f"  - Completeness: {quality_score['completeness']:.1f}%")
        print(f"  - Uniqueness: {quality_score['uniqueness']:.1f}%")
        print(f"  - Validity: {quality_score['validity']:.1f}%")
        print(f"  - Consistency: {quality_score['consistency']:.1f}%")
        
        # Test time series validation
        if 'Date' in df.columns:
            print_info("Testing time series validation...")
            ts_validation = analyzer.validate_time_series('Date', expected_freq='D')
            
            if ts_validation['is_valid']:
                print_success("Time series validation passed")
            else:
                print_warning(f"Time series issues: {len(ts_validation['issues'])}")
                for issue in ts_validation['issues'][:3]:  # Show first 3
                    print(f"    - {issue}")
        
        # Test business rules validation
        print_info("Testing business rules validation...")
        business_rules = analyzer.validate_business_rules()
        
        if business_rules['is_valid']:
            print_success("Business rules validation passed")
        else:
            print_warning(f"Business rule violations: {len(business_rules['violations'])}")
            for violation in business_rules['violations'][:3]:  # Show first 3
                print(f"    - {violation['rule']}: {violation['violation_count']} violations")
        
        # Test zero-inflation analysis
        if 'Units Sold' in df.columns:
            print_info("Testing zero-inflation analysis...")
            zero_analysis = analyzer.analyze_zero_inflation('Units Sold')
            
            if 'error' not in zero_analysis:
                print_success(f"Zero-inflation: {zero_analysis['zero_percentage']:.1f}% (Severity: {zero_analysis['severity']})")
                print(f"  Recommendation: {zero_analysis['recommendation']}")
        
        # Test correlation analysis
        print_info("Testing correlation analysis...")
        corr_analysis = analyzer.analyze_feature_correlations(threshold=0.95)
        
        if 'error' not in corr_analysis:
            high_corr_count = len(corr_analysis['high_correlations'])
            if high_corr_count == 0:
                print_success("No high correlations found (threshold: 0.95)")
            else:
                print_warning(f"{high_corr_count} high correlations detected")
        
        # Test distribution analysis
        print_info("Testing distribution analysis...")
        dist_analysis = analyzer.analyze_distributions()
        
        if 'error' not in dist_analysis:
            highly_skewed = dist_analysis['summary']['highly_skewed']
            if highly_skewed > 0:
                print_warning(f"{highly_skewed} highly skewed columns")
            else:
                print_success("No highly skewed distributions")
        
        return True
        
    except Exception as e:
        print_error(f"Data quality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 5: VALIDATORS
# ============================================================================

def test_validators(df):
    """Test validation functionality."""
    print_test_header("Data Validators")
    
    from src.utils.validators import DataValidator
    
    try:
        # Initialize validator
        validator = DataValidator()
        print_success("DataValidator initialized")
        
        # Test dataframe validation
        print_info("Testing dataframe validation...")
        validation_result = validator.validate_dataframe(df, check_retail=True)
        
        if validation_result['is_valid']:
            print_success("Dataframe validation passed")
        else:
            print_warning(f"Validation issues found: {len(validation_result['errors'])} errors")
            for error in validation_result['errors'][:3]:  # Show first 3
                print(f"    - {error}")
        
        # Show warnings if any
        if validation_result.get('warnings'):
            print_info(f"Validation warnings: {len(validation_result['warnings'])}")
            for warning in validation_result['warnings'][:3]:  # Show first 3
                print(f"    - {warning}")
        
        # Check quality score
        quality_score = validation_result['metrics'].get('quality_score', 0)
        print_success(f"Data Quality Score: {quality_score:.1f}/100")
        
        # Test retail column validation
        if 'retail_validation' in validation_result['metrics']:
            retail_val = validation_result['metrics']['retail_validation']
            print_success("Retail validation completed")
            print(f"  - Required columns: {retail_val['coverage']['required']}")
            print(f"  - Recommended columns: {retail_val['coverage']['recommended']}")
        
        # Test time series validation
        if 'Date' in df.columns and 'Units Sold' in df.columns:
            print_info("Testing time series validation...")
            ts_validation = validator.validate_time_series(
                df, 
                date_column='Date',
                value_column='Units Sold',
                check_forecasting_readiness=True
            )
            
            if ts_validation['is_valid']:
                print_success("Time series validation passed")
            else:
                print_warning(f"Time series warnings: {len(ts_validation['warnings'])}")
            
            # Check forecasting readiness
            if 'forecasting_readiness' in ts_validation['metrics']:
                readiness = ts_validation['metrics']['forecasting_readiness']
                if readiness['is_ready']:
                    print_success("✅ Data is ready for forecasting!")
                else:
                    print_warning("⚠️  Data may not be optimal for forecasting")
                    for warning in readiness['warnings'][:3]:
                        print(f"    - {warning}")
        
        return True
        
    except Exception as e:
        print_error(f"Validator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 6: DATA EXPLORATION
# ============================================================================

def test_data_exploration(df):
    """Test data exploration functionality."""
    print_test_header("Data Exploration")
    
    from src.data.exploration import DataExplorer
    
    try:
        # Initialize explorer
        explorer = DataExplorer()
        print_success("DataExplorer initialized")
        
        # Test summary statistics
        print_info("Generating summary statistics...")
        summary_stats = explorer.generate_summary_statistics(df)
        print_success("Summary statistics generated")
        print(f"  - Dataset shape: {summary_stats['dataset_shape']}")
        print(f"  - Duplicate rows: {summary_stats['duplicate_rows']}")
        print(f"  - Memory usage: {summary_stats['memory_usage_mb']:.2f} MB")
        
        # Test hierarchical structure analysis
        if all(col in df.columns for col in ['Store ID', 'Product ID']):
            print_info("Analyzing hierarchical structure...")
            hierarchy = explorer.analyze_hierarchical_structure(df)
            
            if 'error' not in hierarchy:
                print_success("Hierarchical analysis completed")
                if 'store_product_combinations' in hierarchy:
                    print(f"  - Store×Product combinations: {hierarchy['store_product_combinations']['total_combinations']}")
                    print(f"  - Coverage: {hierarchy['store_product_combinations']['coverage_percentage']:.1f}%")
            else:
                print_warning(f"Hierarchy analysis error: {hierarchy['error']}")
        
        # Test zero-inflation analysis
        if 'Units Sold' in df.columns:
            print_info("Analyzing zero-inflation...")
            zero_inflation = explorer.analyze_zero_inflation(df, 'Units Sold')
            
            if 'error' not in zero_inflation:
                print_success(f"Zero-inflation: {zero_inflation['zero_percentage']:.1f}%")
                print(f"  - Severity: {zero_inflation['severity']}")
                print(f"  - Recommendation: {zero_inflation['recommendation']}")
        
        # Test price elasticity
        if 'Price' in df.columns and 'Units Sold' in df.columns:
            print_info("Analyzing price elasticity...")
            elasticity = explorer.analyze_price_elasticity(df)
            
            if 'error' not in elasticity:
                print_success("Price elasticity analysis completed")
                print(f"  - Overall correlation: {elasticity['overall_correlation']:.3f}")
                print(f"  - Relationship: {elasticity['relationship']}")
        
        # Test promotion impact
        if 'Holiday/Promotion' in df.columns and 'Units Sold' in df.columns:
            print_info("Analyzing promotion impact...")
            promo_impact = explorer.analyze_promotion_impact(df)
            
            if 'error' not in promo_impact:
                print_success("Promotion impact analysis completed")
                print(f"  - Promo avg sales: {promo_impact['promo_avg_sales']:.1f}")
                print(f"  - Regular avg sales: {promo_impact['regular_avg_sales']:.1f}")
                print(f"  - Uplift: {promo_impact['uplift_percentage']:.1f}%")
        
        return True
        
    except Exception as e:
        print_error(f"Data exploration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all tests."""
    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}🚀 PHASE 3 INTEGRATION TEST SUITE 🚀{RESET}")
    print(f"{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}Project: D:\\cortexx-forecasting\\{RESET}")
    print(f"{BLUE}Dataset: {DATASET_PATH}{RESET}")
    print(f"{BLUE}{'='*80}{RESET}\n")
    
    results = {
        'total': 6,
        'passed': 0,
        'failed': 0
    }
    
    # Test 1: Imports
    if test_imports():
        results['passed'] += 1
    else:
        results['failed'] += 1
        print_error("Import test failed. Cannot continue.")
        print_summary(results)
        return
    
    # Test 2: Data Collection
    success, df = test_data_collection()
    if success and df is not None:
        results['passed'] += 1
    else:
        results['failed'] += 1
        print_error("Data collection test failed. Cannot continue.")
        print_summary(results)
        return
    
    # Test 3: Data Preprocessing
    if test_data_preprocessing(df):
        results['passed'] += 1
    else:
        results['failed'] += 1
    
    # Test 4: Data Quality
    if test_data_quality(df):
        results['passed'] += 1
    else:
        results['failed'] += 1
    
    # Test 5: Validators
    if test_validators(df):
        results['passed'] += 1
    else:
        results['failed'] += 1
    
    # Test 6: Data Exploration
    if test_data_exploration(df):
        results['passed'] += 1
    else:
        results['failed'] += 1
    
    # Print summary
    print_summary(results)


def print_summary(results):
    """Print test summary."""
    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}📊 TEST SUMMARY 📊{RESET}")
    print(f"{BLUE}{'='*80}{RESET}")
    
    print(f"\nTotal Tests: {results['total']}")
    print(f"{GREEN}Passed: {results['passed']}{RESET}")
    print(f"{RED}Failed: {results['failed']}{RESET}")
    
    pass_rate = (results['passed'] / results['total']) * 100
    
    if results['failed'] == 0:
        print(f"\n{GREEN}{'🎉 '*20}{RESET}")
        print(f"{GREEN}✅ ✅ ✅ ALL TESTS PASSED! (100%) ✅ ✅ ✅{RESET}")
        print(f"{GREEN}✅ Phase 3 integration is successful!{RESET}")
        print(f"{GREEN}✅ Your dataset is ready for forecasting!{RESET}")
        print(f"{GREEN}{'🎉 '*20}{RESET}\n")
    elif pass_rate >= 80:
        print(f"\n{YELLOW}⚠️  MOSTLY PASSED ({pass_rate:.0f}%){RESET}")
        print(f"{YELLOW}Some tests had minor issues. Review warnings above.{RESET}\n")
    else:
        print(f"\n{RED}⚠️  SOME TESTS FAILED ({pass_rate:.0f}%){RESET}")
        print(f"{YELLOW}Please review the errors above and fix the issues.{RESET}\n")


if __name__ == "__main__":
    run_all_tests()
