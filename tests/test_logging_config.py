"""Tests for structured logging configuration."""

import json
import logging
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import StringIO
import tempfile

from src.core.logging_config import (
    JSONFormatter,
    StructuredLogger,
    setup_logging,
    log_execution,
    get_logger,
    log_model_training,
    log_prediction,
    log_data_validation,
    log_api_request,
    log_model_evaluation,
    log_feature_importance
)


# Fixtures
@pytest.fixture
def temp_log_dir():
    """Create temporary directory for logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
        # Clean up any open file handlers and reset module-level logger
        import src.core.logging_config
        src.core.logging_config._logger = None
        
        for logger_name in list(logging.root.manager.loggerDict.keys()):
            logger = logging.getLogger(logger_name)
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)


@pytest.fixture
def json_formatter():
    """Create JSON formatter instance."""
    return JSONFormatter()


@pytest.fixture
def test_logger():
    """Create test logger."""
    logger = logging.getLogger('test_verdict')
    logger.handlers.clear()
    return logger


# JSONFormatter Tests
class TestJSONFormatter:
    
    def test_format_basic_log(self, json_formatter, test_logger):
        """Test formatting basic log record."""
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='test.py',
            lineno=10,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        formatted = json_formatter.format(record)
        parsed = json.loads(formatted)
        
        assert parsed['message'] == 'Test message'
        assert parsed['level'] == 'INFO'
        assert 'timestamp' in parsed
        assert parsed['line'] == 10
    
    def test_format_with_extra_fields(self, json_formatter, test_logger):
        """Test formatting with custom extra fields."""
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='test.py',
            lineno=20,
            msg='Message with context',
            args=(),
            exc_info=None
        )
        record.extra_fields = {
            'user_id': 123,
            'operation': 'test_op',
            'duration_ms': 45.5
        }
        
        formatted = json_formatter.format(record)
        parsed = json.loads(formatted)
        
        assert parsed['user_id'] == 123
        assert parsed['operation'] == 'test_op'
        assert parsed['duration_ms'] == 45.5
    
    def test_format_log_levels(self, json_formatter):
        """Test formatting different log levels."""
        levels = [
            (logging.DEBUG, 'DEBUG'),
            (logging.INFO, 'INFO'),
            (logging.WARNING, 'WARNING'),
            (logging.ERROR, 'ERROR'),
            (logging.CRITICAL, 'CRITICAL')
        ]
        
        for level, level_name in levels:
            record = logging.LogRecord(
                name='test', level=level, pathname='test.py',
                lineno=1, msg='Test', args=(), exc_info=None
            )
            formatted = json_formatter.format(record)
            parsed = json.loads(formatted)
            assert parsed['level'] == level_name


# StructuredLogger Tests
class TestStructuredLogger:
    
    def test_initialization(self, test_logger):
        """Test StructuredLogger initialization."""
        structured = StructuredLogger(test_logger)
        assert structured.logger == test_logger
    
    def test_debug_logging(self, test_logger):
        """Test debug level logging."""
        structured = StructuredLogger(test_logger)
        with patch.object(structured.logger, 'handle') as mock_handle:
            structured.debug('Debug message', user='test')
            mock_handle.assert_called_once()
    
    def test_info_logging(self, test_logger):
        """Test info level logging."""
        structured = StructuredLogger(test_logger)
        with patch.object(structured.logger, 'handle') as mock_handle:
            structured.info('Info message', operation='test')
            mock_handle.assert_called_once()
    
    def test_warning_logging(self, test_logger):
        """Test warning level logging."""
        structured = StructuredLogger(test_logger)
        with patch.object(structured.logger, 'handle') as mock_handle:
            structured.warning('Warning message', issue='test')
            mock_handle.assert_called_once()
    
    def test_error_logging(self, test_logger):
        """Test error level logging."""
        structured = StructuredLogger(test_logger)
        with patch.object(structured.logger, 'handle') as mock_handle:
            structured.error('Error message', error_code=500)
            mock_handle.assert_called_once()
    
    def test_critical_logging(self, test_logger):
        """Test critical level logging."""
        structured = StructuredLogger(test_logger)
        with patch.object(structured.logger, 'handle') as mock_handle:
            structured.critical('Critical message', severity='high')
            mock_handle.assert_called_once()


# Setup Logging Tests
class TestSetupLogging:
    
    def test_setup_logging_console_only(self, temp_log_dir):
        """Test setup with console logging only."""
        logger = setup_logging(
            log_dir=temp_log_dir,
            enable_console=True,
            enable_file=False
        )
        
        assert logger is not None
        assert isinstance(logger, StructuredLogger)
        assert len(logger.logger.handlers) == 1
    
    def test_setup_logging_file_only(self, temp_log_dir):
        """Test setup with file logging only."""
        logger = setup_logging(
            log_dir=temp_log_dir,
            enable_console=False,
            enable_file=True
        )
        
        assert logger is not None
        assert len(logger.logger.handlers) == 1
    
    def test_setup_logging_both(self, temp_log_dir):
        """Test setup with both console and file logging."""
        logger = setup_logging(
            log_dir=temp_log_dir,
            enable_console=True,
            enable_file=True
        )
        
        assert logger is not None
        assert len(logger.logger.handlers) == 2
    
    def test_setup_logging_creates_directory(self, temp_log_dir):
        """Test that setup creates log directory if needed."""
        new_log_dir = Path(temp_log_dir) / 'new_logs'
        logger = setup_logging(
            log_dir=str(new_log_dir),
            enable_file=True
        )
        
        assert new_log_dir.exists()


# Log Execution Decorator Tests
class TestLogExecution:
    
    def test_log_execution_success(self, temp_log_dir):
        """Test log_execution decorator on success."""
        @log_execution('test_operation')
        def test_func():
            return 42
        
        with patch('src.core.logging_config.setup_logging') as mock_setup:
            mock_logger = MagicMock()
            mock_setup.return_value = mock_logger
            
            result = test_func()
            
            assert result == 42
            # Should log started and completed
            assert mock_logger.info.call_count >= 1
    
    def test_log_execution_with_error(self, temp_log_dir):
        """Test log_execution decorator on error."""
        @log_execution('failing_operation')
        def test_func():
            raise ValueError('Test error')
        
        with patch('src.core.logging_config.setup_logging') as mock_setup:
            mock_logger = MagicMock()
            mock_setup.return_value = mock_logger
            
            with pytest.raises(ValueError):
                test_func()
            
            # Should log error
            assert mock_logger.error.called
    
    def test_log_execution_log_input(self, temp_log_dir):
        """Test log_execution with input logging."""
        @log_execution('operation', log_input=True)
        def test_func(a, b, c=None):
            return a + b
        
        with patch('src.core.logging_config.setup_logging') as mock_setup:
            mock_logger = MagicMock()
            mock_setup.return_value = mock_logger
            
            result = test_func(1, 2, c=3)
            assert result == 3
    
    def test_log_execution_log_output(self, temp_log_dir):
        """Test log_execution with output logging."""
        @log_execution('operation', log_output=True)
        def test_func():
            return {'result': 'data'}
        
        with patch('src.core.logging_config.setup_logging') as mock_setup:
            mock_logger = MagicMock()
            mock_setup.return_value = mock_logger
            
            result = test_func()
            assert result == {'result': 'data'}


# Convenience Logging Functions Tests
class TestConvenienceLoggingFunctions:
    
    @patch('src.core.logging_config.get_logger')
    def test_log_model_training(self, mock_get_logger):
        """Test model training logging."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        log_model_training(
            model_name='RandomForest',
            dataset_rows=1000,
            features_count=10,
            target_column='target',
            cv_folds=5
        )
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert 'Model training started' in str(call_args)
    
    @patch('src.core.logging_config.get_logger')
    def test_log_prediction(self, mock_get_logger):
        """Test prediction logging."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        log_prediction(
            model_name='LogisticRegression',
            prediction_value=1,
            confidence=0.95,
            input_features_count=10,
            prediction_time_ms=45.2
        )
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert 'Prediction made' in str(call_args)
    
    @patch('src.core.logging_config.get_logger')
    def test_log_data_validation(self, mock_get_logger):
        """Test data validation logging."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        log_data_validation(
            validation_type='format',
            dataset_size=500,
            is_valid=True,
            issues_found=0
        )
        
        mock_logger.info.assert_called_once()
    
    @patch('src.core.logging_config.get_logger')
    def test_log_api_request(self, mock_get_logger):
        """Test API request logging."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        log_api_request(
            endpoint='/api/predict',
            method='POST',
            status_code=200,
            response_time_ms=125.5,
            user_id='user123'
        )
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert 'API request' in str(call_args)
    
    @patch('src.core.logging_config.get_logger')
    def test_log_model_evaluation(self, mock_get_logger):
        """Test model evaluation logging."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        metrics = {
            'accuracy': 0.95,
            'precision': 0.93,
            'recall': 0.91,
            'f1': 0.92
        }
        
        log_model_evaluation(
            model_name='XGBoost',
            metrics=metrics,
            dataset_size=2000
        )
        
        mock_logger.info.assert_called_once()
    
    @patch('src.core.logging_config.get_logger')
    def test_log_feature_importance(self, mock_get_logger):
        """Test feature importance logging."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        log_feature_importance(
            model_name='RandomForest',
            features_analyzed=15,
            top_feature_name='income',
            top_feature_importance=0.285
        )
        
        mock_logger.info.assert_called_once()


# Integration Tests
class TestLoggingIntegration:
    
    def test_end_to_end_logging(self, temp_log_dir):
        """Test complete logging workflow."""
        logger = setup_logging(
            log_dir=temp_log_dir,
            enable_console=True,
            enable_file=True
        )
        
        # Log various operations
        logger.info('Application started', version='1.0')
        logger.debug('Debug info', user_id=123)
        logger.warning('Low memory', available_mb=256)
        
        # Verify logger is working
        assert logger is not None
    
    def test_get_logger_singleton(self, temp_log_dir):
        """Test that get_logger returns singleton."""
        with patch('src.core.logging_config.setup_logging') as mock_setup:
            mock_logger1 = MagicMock()
            mock_setup.return_value = mock_logger1
            
            logger1 = get_logger()
            
            mock_setup.reset_mock()
            mock_logger2 = MagicMock()
            mock_setup.return_value = mock_logger2
            
            logger2 = get_logger()
            
            # Should return the same instance
            assert logger1 is logger2
