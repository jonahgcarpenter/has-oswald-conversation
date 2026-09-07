"""Shared fixtures for tests against real Home Assistant APIs."""

import pytest


@pytest.fixture(autouse=True)
def custom_integrations(enable_custom_integrations):
    """Allow Home Assistant to discover this custom integration."""
