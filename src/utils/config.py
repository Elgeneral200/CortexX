"""
Enterprise Configuration Management for CortexX Forecasting Platform.

Handles environment-specific configurations, secrets management, and application settings.
Supports development, staging, and production environments.
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
import yaml
from dataclasses import dataclass, field

# Load environment variables
load_dotenv()


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    host: str = os.getenv('DB_HOST', 'localhost')
    port: int = int(os.getenv('DB_PORT', '5432'))
    name: str = os.getenv('DB_NAME', 'cortexx_forecasting')
    user: str = os.getenv('DB_USER', 'admin')
    password: str = os.getenv('DB_PASSWORD', '')
    max_connections: int = 20
    connection_timeout: int = 30


@dataclass
class ModelConfig:
    """Model training and deployment configuration."""
    default_test_size: float = 0.2
    default_random_state: int = 42
    max_training_time: int = 3600  # seconds
    model_registry_path: str = os.getenv('MODEL_REGISTRY_PATH', './models/registry')
    checkpoint_dir: str = './models/checkpoints'
    supported_models: list = field(default_factory=lambda: [
        'XGBoost', 'LightGBM', 'Random Forest', 'CatBoost',
        'Linear Regression', 'Ridge Regression', 'Lasso Regression',
        'Decision Tree', 'K-Nearest Neighbors', 'Support Vector Regression', 'Prophet'
    ])
    enable_gpu: bool = os.getenv('ENABLE_GPU', 'false').lower() == 'true'


@dataclass
class OptimizationConfig:
    """Hyperparameter optimization configuration."""
    default_n_trials: int = 50
    default_cv_splits: int = 3
    optimization_timeout: int = 7200  # seconds
    parallel_jobs: int = -1
    sampler_type: str = 'TPE'  # Tree-structured Parzen Estimator


@dataclass
class BacktestingConfig:
    """Backtesting configuration."""
    initial_train_size: int = 100
    test_size: int = 30
    step_size: int = 30
    window_type: str = 'expanding'  # or 'rolling'
    enable_parallel: bool = True


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = os.getenv('API_HOST', '0.0.0.0')
    port: int = int(os.getenv('API_PORT', '8000'))
    debug: bool = os.getenv('DEBUG', 'false').lower() == 'true'
    workers: int = int(os.getenv('WORKERS', '4'))
    timeout: int = 300
    max_request_size: int = 100 * 1024 * 1024  # 100MB
    cors_origins: list = field(default_factory=lambda: ['*'])


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = os.getenv('LOG_LEVEL', 'INFO')
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_dir: str = './logs'
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_console: bool = True
    enable_file: bool = True


@dataclass
class SecurityConfig:
    """Security and authentication configuration."""
    secret_key: str = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')
    api_key: str = os.getenv('API_KEY', '')
    jwt_algorithm: str = 'HS256'
    jwt_expiration: int = 3600  # seconds
    enable_authentication: bool = os.getenv('ENABLE_AUTH', 'false').lower() == 'true'
    allowed_ips: list = field(default_factory=list)


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration."""
    enable_monitoring: bool = os.getenv('ENABLE_MONITORING', 'true').lower() == 'true'
    metrics_port: int = int(os.getenv('METRICS_PORT', '9090'))
    alert_email: str = os.getenv('ALERT_EMAIL', '')
    alert_threshold_rmse: float = 100.0
    drift_detection_threshold: float = 0.1
    performance_degradation_threshold: float = 0.2


@dataclass
class CacheConfig:
    """Caching configuration."""
    enable_cache: bool = os.getenv('ENABLE_CACHE', 'true').lower() == 'true'
    cache_backend: str = 'memory'  # or 'redis'
    redis_host: str = os.getenv('REDIS_HOST', 'localhost')
    redis_port: int = int(os.getenv('REDIS_PORT', '6379'))
    cache_ttl: int = 3600  # seconds
    max_cache_size: int = 1000


class Config:
    """
    Central configuration class for CortexX Enterprise Platform.
    
    Manages all application settings and provides environment-specific configurations.
    """
    
    def __init__(self, environment: str = None):
        """
        Initialize configuration.
        
        Args:
            environment: Environment name (development, staging, production)
        """
        self.environment = environment or os.getenv('ENVIRONMENT', 'development')
        self.debug = self.environment == 'development'
        
        # Initialize all configuration sections
        self.database = DatabaseConfig()
        self.model = ModelConfig()
        self.optimization = OptimizationConfig()
        self.backtesting = BacktestingConfig()
        self.api = APIConfig()
        self.logging = LoggingConfig()
        self.security = SecurityConfig()
        self.monitoring = MonitoringConfig()
        self.cache = CacheConfig()
        
        # Create necessary directories
        self._create_directories()
        
        # Setup logging
        self._setup_logging()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.model.model_registry_path,
            self.model.checkpoint_dir,
            self.logging.log_dir,
            './data/raw',
            './data/processed',
            './data/predictions',
            './reports',
            './artifacts'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.logging.level.upper(), logging.INFO)
        
        # Create formatters
        formatter = logging.Formatter(self.logging.format)
        
        # Root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Remove existing handlers
        root_logger.handlers = []
        
        # Console handler
        if self.logging.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # File handler with rotation
        if self.logging.enable_file:
            from logging.handlers import RotatingFileHandler
            log_file = Path(self.logging.log_dir) / 'cortexx.log'
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=self.logging.max_bytes,
                backupCount=self.logging.backup_count
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        logging.info(f"CortexX Platform initialized in {self.environment} environment")
    
    def load_from_yaml(self, config_path: str):
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Update configuration from YAML
            for section, values in config_data.items():
                if hasattr(self, section):
                    section_config = getattr(self, section)
                    for key, value in values.items():
                        if hasattr(section_config, key):
                            setattr(section_config, key, value)
            
            logging.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logging.error(f"Failed to load configuration from {config_path}: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'environment': self.environment,
            'database': self.database.__dict__,
            'model': self.model.__dict__,
            'optimization': self.optimization.__dict__,
            'backtesting': self.backtesting.__dict__,
            'api': self.api.__dict__,
            'logging': self.logging.__dict__,
            'security': {k: v for k, v in self.security.__dict__.items() 
                        if k not in ['secret_key', 'api_key', 'jwt_secret']},  # Exclude secrets
            'monitoring': self.monitoring.__dict__,
            'cache': self.cache.__dict__
        }
    
    def validate(self) -> bool:
        """
        Validate configuration settings.
        
        Returns:
            bool: True if configuration is valid
        """
        errors = []
        
        # Validate database
        if not self.database.host:
            errors.append("Database host is required")
        
        # Validate security in production
        if self.environment == 'production':
            if self.security.secret_key == 'your-secret-key-change-in-production':
                errors.append("SECRET_KEY must be changed in production")
            if not self.security.enable_authentication:
                errors.append("Authentication should be enabled in production")
        
        # Validate model paths
        if not Path(self.model.model_registry_path).exists():
            errors.append(f"Model registry path does not exist: {self.model.model_registry_path}")
        
        if errors:
            for error in errors:
                logging.error(f"Configuration validation error: {error}")
            return False
        
        logging.info("Configuration validation passed")
        return True
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get model-specific configuration.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict containing model configuration
        """
        base_config = {
            'random_state': self.model.default_random_state,
            'n_jobs': self.optimization.parallel_jobs
        }
        
        # Model-specific defaults
        model_configs = {
            'XGBoost': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
            'LightGBM': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
            'Random Forest': {'n_estimators': 100, 'max_depth': 10},
            'CatBoost': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
            'Ridge Regression': {'alpha': 1.0},
            'Lasso Regression': {'alpha': 1.0},
            'Decision Tree': {'max_depth': 10},
            'K-Nearest Neighbors': {'n_neighbors': 5},
            'Support Vector Regression': {'C': 1.0, 'epsilon': 0.1}
        }
        
        config = base_config.copy()
        if model_name in model_configs:
            config.update(model_configs[model_name])
        
        return config


# Global configuration instance
_config_instance: Optional[Config] = None


def get_config(environment: str = None) -> Config:
    """
    Get or create global configuration instance.
    
    Args:
        environment: Environment name (optional)
        
    Returns:
        Config: Global configuration instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config(environment)
    
    return _config_instance


def reset_config():
    """Reset global configuration instance (mainly for testing)."""
    global _config_instance
    _config_instance = None


# Constants for easy access
SUPPORTED_MODELS = [
    'XGBoost', 'LightGBM', 'Random Forest', 'CatBoost',
    'Linear Regression', 'Ridge Regression', 'Lasso Regression',
    'Decision Tree', 'K-Nearest Neighbors', 'Support Vector Regression', 'Prophet'
]

SUPPORTED_METRICS = ['rmse', 'mae', 'r2', 'mape', 'mse']

DEFAULT_HYPERPARAMETER_SPACES = {
    'xgboost': {
        'n_estimators': (50, 300),
        'max_depth': (3, 10),
        'learning_rate': (0.01, 0.3),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0)
    },
    'lightgbm': {
        'n_estimators': (50, 300),
        'max_depth': (3, 10),
        'learning_rate': (0.01, 0.3),
        'num_leaves': (20, 150),
        'subsample': (0.6, 1.0)
    },
    'random_forest': {
        'n_estimators': (50, 300),
        'max_depth': (5, 30),
        'min_samples_split': (2, 20),
        'min_samples_leaf': (1, 10)
    }
}
