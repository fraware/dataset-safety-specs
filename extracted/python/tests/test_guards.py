# Python tests for generated guards

import pytest
from ds_guard import Row, phi_guard, coppa_guard, gdpr_guard

def test_phi_guard():
    """Test Detects Protected Health Information."""
    row = Row(phi=["SSN: 123-45-6789"])
    assert phi_guard(row)

def test_coppa_guard():
    """Test Detects COPPA violations (age < 13)."""
    row = Row(phi=["SSN: 123-45-6789"])
    assert coppa_guard(row)

def test_gdpr_guard():
    """Test Detects GDPR special categories."""
    row = Row(phi=["SSN: 123-45-6789"])
    assert gdpr_guard(row)

def test_clean_row():
    """Test that clean rows pass all guards."""
    row = Row(age=25)
    assert not phi_guard(row)
    assert not coppa_guard(row)
    assert not gdpr_guard(row)