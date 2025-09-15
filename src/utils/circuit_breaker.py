# src/utils/circuit_breaker.py
import time
import threading
from enum import Enum
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass
from src.utils.logger import setup_logger


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    timeout: float = 60.0
    expected_exception: type = Exception
    fallback: Optional[Callable] = None
    name: str = "circuit_breaker"


class CircuitBreakerError(Exception):
    """Circuit breaker is open"""
    pass


class CircuitBreaker:
    """Circuit breaker pattern implementation"""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED
        self.lock = threading.RLock()
        self.logger = setup_logger(f"CircuitBreaker.{config.name}")

    def __call__(self, func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)

        return wrapper

    def call(self, func: Callable, *args, **kwargs) -> Any:
        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.logger.info(
                        "Circuit breaker transitioning to HALF_OPEN",
                    )
                else:
                    if self.config.fallback:
                        self.logger.info("Circuit OPEN, using fallback")
                        return self.config.fallback(*args, **kwargs)
                    raise CircuitBreakerError(f"Circuit {self.config.name} is OPEN")

        try:
            result = func(*args, **kwargs)
            with self.lock:
                if self.state == CircuitState.HALF_OPEN:
                    self.state = CircuitState.CLOSED
                    self.logger.info("Circuit breaker transitioning to CLOSED")
                self.failure_count = 0
                self.last_failure_time = None
            return result
        except self.config.expected_exception as e:
            with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitState.OPEN
                    self.logger.warning(
                        "Circuit breaker transitioning to OPEN",
                        extra={"failure_count": self.failure_count, "error": str(e)},
                    )
            if self.config.fallback:
                self.logger.info("Using fallback due to failure")
                return self.config.fallback(*args, **kwargs)
            raise

    def _should_attempt_reset(self) -> bool:
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) >= self.config.timeout

    @property
    def status(self) -> Dict[str, Any]:
        return {
            "name": self.config.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "failure_threshold": self.config.failure_threshold,
            "timeout": self.config.timeout,
        }
