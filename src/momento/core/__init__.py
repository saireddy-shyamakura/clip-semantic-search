"""
core — Foundational infrastructure for Momento.

Provides configuration, device management, process locking,
shutdown handling, logging, validation, and error types.
"""
from .config import load_config, save_config, apply_config_overrides, MomentoConfig, get_device
from .device import DeviceManager, device_manager
from .lock import LockFile
from .shutdown import is_shutdown_requested, install_signal_handlers, reset_shutdown_flag
from .logger import set_log_format, get_logger, MetricLogger, JsonFormatter
from .validation import validate_image_path, validate_text_query, validate_folder_path, validate_positive_int, validate_choice, ValidationError
from .exceptions import FeatureError, ValidationError as HandlerValidationError

__all__ = [
    "load_config", "save_config", "apply_config_overrides", "MomentoConfig", "get_device",
    "DeviceManager", "device_manager",
    "LockFile",
    "is_shutdown_requested", "install_signal_handlers", "reset_shutdown_flag",
    "set_log_format", "get_logger", "MetricLogger", "JsonFormatter",
    "validate_image_path", "validate_text_query", "validate_folder_path",
    "validate_positive_int", "validate_choice", "ValidationError",
    "FeatureError",
]