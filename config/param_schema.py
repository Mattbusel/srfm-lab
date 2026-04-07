"""
config/param_schema.py
======================
Re-exports ParamSchema and SchemaValidationError from param_manager
for use by optimization modules that import from this module path.
"""

from config.param_manager import ParamSchema, SchemaValidationError  # noqa: F401

__all__ = ["ParamSchema", "SchemaValidationError"]
