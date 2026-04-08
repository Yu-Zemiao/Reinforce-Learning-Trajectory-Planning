import atexit
import logging
import queue
import sys
import threading
import ctypes
from pathlib import Path
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler


class _ColorFormatter(logging.Formatter):
    _RESET = "\033[0m"
    _COLORS = {
        logging.DEBUG: "\033[31m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[35m",
        logging.CRITICAL: "\033[41m\033[97m",
    }

    def format(self, record):
        base_message = super().format(record)
        color = self._COLORS.get(record.levelno)
        if not color:
            return base_message
        parts = base_message.split(" | ", 3)
        if len(parts) < 4:
            return f"{color}{base_message}{self._RESET}"
        prefix = f"{parts[0]} | {parts[1]} | {parts[2]} | "
        message = parts[3]
        return f"{color}{prefix}{self._RESET}{message}"


def _enable_windows_ansi():
    if sys.platform != "win32":
        return
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


class _RootAsyncLoggerSingleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._log_dir = Path(__file__).resolve().parents[1] / "log"
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_dir / "run.log"

        self._queue: queue.Queue = queue.Queue(maxsize=10000)
        self._queue_handler = QueueHandler(self._queue)
        self._queue_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        color_formatter = _ColorFormatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        _enable_windows_ansi()

        self._stream_handler = logging.StreamHandler(sys.stdout)
        self._stream_handler.setLevel(logging.INFO)
        self._stream_handler.setFormatter(color_formatter)

        # 使用 RotatingFileHandler 限制日志文件大小为 20MB
        # 当日志文件达到 20MB 时会自动轮转，保留最近 5 个备份文件
        self._file_handler = RotatingFileHandler(
            filename=self._log_file,
            maxBytes=15 * 1024 * 1024,  # 20MB
            backupCount=3,
            encoding="utf-8"
        )
        self._file_handler.setLevel(logging.DEBUG)
        self._file_handler.setFormatter(formatter)

        self._listener = QueueListener(
            self._queue,
            self._stream_handler,
            self._file_handler,
            respect_handler_level=True,
        )
        self._listener.start()

        self._root_logger = logging.getLogger()
        self._root_logger.setLevel(logging.DEBUG)
        self._root_logger.handlers.clear()
        self._root_logger.addHandler(self._queue_handler)

        atexit.register(self.shutdown)
        self._initialized = True

    def get_root_logger(self) -> logging.Logger:
        return self._root_logger

    def shutdown(self):
        if hasattr(self, "_listener") and self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None


_LOGGER_SINGLETON = _RootAsyncLoggerSingleton()


def setup_root_logger() -> logging.Logger:
    return _LOGGER_SINGLETON.get_root_logger()


def get_logger(name=None) -> logging.Logger:
    if name:
        return logging.getLogger(name)
    return setup_root_logger()


def shutdown_logger():
    _LOGGER_SINGLETON.shutdown()


logger = setup_root_logger()
