"""Re-export create_fixtures for package backwards compatibility."""
from data.fixtures.create_fixtures import create_test_db

__all__ = ["create_test_db"]
