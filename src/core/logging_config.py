"""Structured logging configuration for VERDICT platform."""

import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from functools import wraps

import sys


class JSONFormatter(logging.Formatter):
    """Custom formatter for JSON-structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Include exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Include custom fields
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data)


class StructuredLogger:
    """Wrapper for structured logging with custom fields."""

    def __init__(self, logger: logging.Logger):
        """Initialize structured logger.
        
        Args:
            logger: Base Python logger instance
        """
        self.logger = logger

    def log_with_context(self, level: int, message: str, **kwargs) -> None:
        """Log message with additional context fields.
        
        Args:
            level: Logging level
            message: Log message
            **kwargs: Additional context fields to include in JSON
        """
        record = self.logger.makeRecord(
            self.logger.name, level, '', 0, message, (), None
        )
        record.extra_fields = kwargs
        self.logger.handle(record)

    def debug(self, message: str, **kwargs) -> None:
        """Log at DEBUG level."""
        self.log_with_context(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log at INFO level."""
        self.log_with_context(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log at WARNING level."""
        self.log_with_context(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log at ERROR level."""
        self.log_with_context(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """Log at CRITICAL level."""
        self.log_with_context(logging.CRITICAL, message, **kwargs)


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    enable_console: bool = True,
    enable_file: bool = True
) -> StructuredLogger:
    """Setup structured logging with JSON formatter.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_console: Whether to log to console
        enable_file: Whether to log to files
        
    Returns:
        StructuredLogger instance for application use
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('verdict')
    logger.setLevel(getattr(logging, log_level))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler (JSON)
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level))
        console_handler.setFormatter(JSONFormatter())
        logger.addHandler(console_handler)
    
    # File handler (JSON, rotated daily)
    if enable_file:
        log_file = log_path / f"verdict_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            str(log_file),
            maxBytes=10485760,  # 10MB
            backupCount=7  # Keep 7 days of logs
        )
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)
    
    return StructuredLogger(logger)


def log_execution(
    operation_name: str,
    log_input: bool = True,
    log_output: bool = True,
    log_errors: bool = True
):
    """Decorator to log function execution with timing.
    
    Args:
        operation_name: Human-readable operation name
        log_input: Whether to log input parameters
        log_output: Whether to log return value
        log_errors: Whether to log exceptions
        
    Example:
        @log_execution('model_training', log_input=True)
        def train_model(X, y):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = setup_logging()
            start_time = datetime.now()
            
            # Log start
            context = {
                'operation': operation_name,
                'status': 'started',
                'function': func.__name__
            }
            if log_input:
                context['args_count'] = len(args)
                context['kwargs_keys'] = list(kwargs.keys())
            logger.info(f"{operation_name} started", **context)
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Log completion
                elapsed = (datetime.now() - start_time).total_seconds()
                completion_context = {
                    'operation': operation_name,
                    'status': 'completed',
                    'elapsed_seconds': elapsed
                }
                if log_output:
                    if isinstance(result, (dict, list)):
                        completion_context['result_type'] = type(result).__name__
                        if isinstance(result, dict):
                            completion_context['result_keys'] = list(result.keys())
                        elif isinstance(result, list):
                            completion_context['result_length'] = len(result)
                logger.info(f"{operation_name} completed", **completion_context)
                
                return result
            
            except Exception as e:
                # Log error
                elapsed = (datetime.now() - start_time).total_seconds()
                error_context = {
                    'operation': operation_name,
                    'status': 'failed',
                    'elapsed_seconds': elapsed,
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
                if log_errors:
                    logger.error(f"{operation_name} failed", **error_context)
                raise
        
        return wrapper
    return decorator


# Module-level logger instance
_logger = None


def get_logger() -> StructuredLogger:
    """Get or create the global structured logger.
    
    Returns:
        StructuredLogger instance
    """
    global _logger
    if _logger is None:
        _logger = setup_logging()
    return _logger


# Convenience logging functions
def log_model_training(
    model_name: str,
    dataset_rows: int,
    features_count: int,
    target_column: str,
    **kwargs
) -> None:
    """Log model training initiation.
    
    Args:
        model_name: Name of model being trained
        dataset_rows: Number of training rows
        features_count: Number of features
        target_column: Target column name
        **kwargs: Additional context
    """
    logger = get_logger()
    context = {
        'operation': 'model_training',
        'model_name': model_name,
        'dataset_rows': dataset_rows,
        'features_count': features_count,
        'target_column': target_column
    }
    context.update(kwargs)
    logger.info("Model training started", **context)


def log_prediction(
    model_name: str,
    prediction_value: Any,
    confidence: float,
    input_features_count: int,
    **kwargs
) -> None:
    """Log prediction made by model.
    
    Args:
        model_name: Name of model making prediction
        prediction_value: The predicted value
        confidence: Confidence score (0-1)
        input_features_count: Number of input features
        **kwargs: Additional context
    """
    logger = get_logger()
    context = {
        'operation': 'prediction',
        'model_name': model_name,
        'prediction': str(prediction_value),
        'confidence': float(confidence),
        'input_features_count': input_features_count
    }
    context.update(kwargs)
    logger.info("Prediction made", **context)


def log_data_validation(
    validation_type: str,
    dataset_size: int,
    is_valid: bool,
    issues_found: int = 0,
    **kwargs
) -> None:
    """Log data validation result.
    
    Args:
        validation_type: Type of validation performed
        dataset_size: Size of dataset validated
        is_valid: Whether validation passed
        issues_found: Number of issues detected
        **kwargs: Additional context
    """
    logger = get_logger()
    context = {
        'operation': 'data_validation',
        'validation_type': validation_type,
        'dataset_size': dataset_size,
        'valid': is_valid,
        'issues_found': issues_found
    }
    context.update(kwargs)
    status = "passed" if is_valid else "found_issues"
    logger.info(f"Data validation {status}", **context)


def log_api_request(
    endpoint: str,
    method: str,
    status_code: int,
    response_time_ms: float,
    **kwargs
) -> None:
    """Log API request and response.
    
    Args:
        endpoint: API endpoint path
        method: HTTP method (GET, POST, etc.)
        status_code: HTTP response status code
        response_time_ms: Response time in milliseconds
        **kwargs: Additional context
    """
    logger = get_logger()
    context = {
        'operation': 'api_request',
        'endpoint': endpoint,
        'method': method,
        'status_code': status_code,
        'response_time_ms': float(response_time_ms)
    }
    context.update(kwargs)
    logger.info(f"API request {method} {endpoint}", **context)


def log_model_evaluation(
    model_name: str,
    metrics: Dict[str, float],
    dataset_size: int,
    **kwargs
) -> None:
    """Log model evaluation metrics.
    
    Args:
        model_name: Name of model evaluated
        metrics: Dictionary of metric names and values
        dataset_size: Size of evaluation dataset
        **kwargs: Additional context
    """
    logger = get_logger()
    context = {
        'operation': 'model_evaluation',
        'model_name': model_name,
        'dataset_size': dataset_size
    }
    # Add each metric
    for metric_name, metric_value in metrics.items():
        context[f"metric_{metric_name}"] = float(metric_value)
    context.update(kwargs)
    logger.info("Model evaluation completed", **context)


def log_feature_importance(
    model_name: str,
    features_analyzed: int,
    top_feature_name: str,
    top_feature_importance: float,
    **kwargs
) -> None:
    """Log feature importance analysis.
    
    Args:
        model_name: Name of model analyzed
        features_analyzed: Total features analyzed
        top_feature_name: Name of most important feature
        top_feature_importance: Importance score of top feature
        **kwargs: Additional context
    """
    logger = get_logger()
    context = {
        'operation': 'feature_importance',
        'model_name': model_name,
        'features_analyzed': features_analyzed,
        'top_feature': top_feature_name,
        'top_feature_importance': float(top_feature_importance)
    }
    context.update(kwargs)
    logger.info("Feature importance analysis completed", **context)


# Export public API
__all__ = [
    'setup_logging',
    'StructuredLogger',
    'JSONFormatter',
    'log_execution',
    'get_logger',
    'log_model_training',
    'log_prediction',
    'log_data_validation',
    'log_api_request',
    'log_model_evaluation',
    'log_feature_importance',
]
