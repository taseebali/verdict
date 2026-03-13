"""Tests for formatters module."""

import pytest
from src.core.formatters import format_value


class TestFormatValue:
    """Test format_value function."""

    def test_currency_formatting(self):
        """Test currency formatting for charge-related features."""
        assert format_value('monthlyCharges', 89.50) == '$89.50'
        assert format_value('cost', 100.0) == '$100.00'
        assert format_value('price', 50.25) == '$50.25'
        assert format_value('payment', 999.99) == '$999.99'
        assert format_value('income', 5000.0) == '$5000.00'

    def test_percentage_formatting(self):
        """Test percentage formatting for rate-related features."""
        assert format_value('churnRate', 12.5) == '12.5%'
        assert format_value('percent_change', 100.0) == '100.0%'
        assert format_value('ratio', 0.5) == '0.5%'

    def test_temporal_formatting(self):
        """Test temporal formatting for duration features."""
        assert format_value('tenure', 24) == '24 months'
        assert format_value('months_active', 12) == '12 months'
        assert format_value('age', 35) == '35 years'

    def test_time_unit_formatting(self):
        """Test formatting for time-related features."""
        assert format_value('hours_used', 10.5) == '10.5 hrs'
        assert format_value('usage_hours', 5.0) == '5.0 hrs'
        assert format_value('days_remaining', 7) == '7 days'

    def test_default_formatting(self):
        """Test default formatting for unrecognized features."""
        assert format_value('unknown_feature', 3.14159) == '3.14'
        assert format_value('random_value', 0.001) == '0.00'

    def test_case_insensitivity(self):
        """Test that formatting is case-insensitive."""
        assert format_value('MONTHLY_CHARGES', 89.5) == '$89.50'
        assert format_value('MonthlyCharges', 89.5) == '$89.50'
        assert format_value('ChurnRate', 12.5) == '12.5%'

    def test_edge_cases(self):
        """Test edge cases."""
        assert format_value('charge', 0.0) == '$0.00'
        assert format_value('rate', 0.0) == '0.0%'
        assert format_value('tenure', 0) == '0 months'
        assert format_value('age', 100) == '100 years'
