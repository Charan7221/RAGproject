"""
Advanced Logging Configuration for RAG API
Provides detailed request tracking with session_id and code location
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import traceback
from typing import Optional
from contextvars import ContextVar

# Context variable for session tracking (thread-safe)
session_id_var: ContextVar[Optional[str]] = ContextVar('session_id', default=None)


def set_session_context(session_id: Optional[str]):
    """Set session_id in context for current request."""
    session_id_var.set(session_id)


def get_session_context() -> Optional[str]:
    """Get session_id from context."""
    try:
        return session_id_var.get()
    except LookupError:
        return None


class CustomFormatter(logging.Formatter):
    """Custom formatter with color coding and enhanced format."""
    
    # Color codes
    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    green = "\x1b[38;5;46m"
    cyan = "\x1b[38;5;51m"

    def __init__(self, fmt: str):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset,
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


class SessionContextFilter(logging.Filter):
    """Add session_id context to log records."""
    
    def filter(self, record):
        record.session_id = get_session_context() or "NO_SESSION"
        return True


def setup_logging(
    log_file: str = "logs/rag_api.log",
    log_level: str = "INFO",
    console_output: bool = True
):
    """
    Setup comprehensive logging system.
    
    Args:
        log_file: Path to log file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Whether to output to console
    
    Returns:
        Configured logger instance
    """
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("rag_api")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Format with session_id and location
    # [SESSION_ID][LEVEL][file.function:line] - Message
    log_format = "[%(session_id)s][%(levelname)s][%(filename)s.%(funcName)s:%(lineno)d] - %(message)s"
    
    # File Handler (detailed logs)
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)  # Capture everything in file
    file_formatter = logging.Formatter(
        fmt="%(asctime)s - " + log_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.addFilter(SessionContextFilter())
    logger.addHandler(file_handler)
    
    # Console Handler (with colors)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_formatter = CustomFormatter(log_format)
        console_handler.addFilter(SessionContextFilter())
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str = "rag_api") -> logging.Logger:
    """Get configured logger instance."""
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin to add logger to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger()
    
    def log_method_call(self, method_name: str, **kwargs):
        """Log method call with parameters."""
        self.logger.debug(f"Called {method_name} with params: {kwargs}")
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error with full traceback."""
        self.logger.error(
            f"{context} - Error: {str(error)}\n"
            f"Traceback:\n{traceback.format_exc()}"
        )


# Initialize default logger
default_logger = setup_logging(
    log_file="logs/rag_api.log",
    log_level="DEBUG",  # Capture everything
    console_output=True
)
