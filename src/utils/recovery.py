# src/utils/recovery.py
import time
from typing import Dict, Any, Callable, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import requests
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from src.utils.logger import setup_logger


class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Alert:
    severity: AlertSeverity
    title: str
    message: str
    component: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlertManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logger("AlertManager")
        self.alerts: List[Alert] = []
        self.alert_counts: Dict[str, int] = {}
        self.last_alert_times: Dict[str, float] = {}
        self.throttle_seconds = int(os.getenv("ALERT_THROTTLE_SECONDS", "300"))

    def send_alert(self, alert: Alert) -> None:
        key = f"{alert.component}:{alert.title}"
        now = time.time()
        if key in self.last_alert_times and now - self.last_alert_times[key] < self.throttle_seconds:
            self.alert_counts[key] = self.alert_counts.get(key, 0) + 1
            return
        # append aggregated count if any
        if key in self.alert_counts and self.alert_counts[key] > 0:
            alert.message += f" (Occurred {self.alert_counts[key] + 1} times)"
            self.alert_counts[key] = 0
        self.last_alert_times[key] = now
        self.alerts.append(alert)
        try:
            self._send_slack(alert)
            self._send_email(alert)
            self.logger.info("Alert sent", extra={"severity": alert.severity.value, "title": alert.title})
        except Exception as e:
            self.logger.error("Failed to send alert", extra={"error": str(e)})

    def _send_slack(self, alert: Alert) -> None:
        webhook = os.getenv("SLACK_WEBHOOK_URL", "")
        if not webhook:
            return
        payload = {
            "text": f"[{alert.severity.value.upper()}] {alert.component}: {alert.title}\n{alert.message}",
            "attachments": [
                {
                    "color": "danger" if alert.severity in (AlertSeverity.HIGH, AlertSeverity.CRITICAL) else "warning",
                    "fields": [
                        {"title": "Component", "value": alert.component, "short": True},
                        {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                        {"title": "Metadata", "value": json.dumps(alert.metadata)[:1000], "short": False},
                    ],
                }
            ],
        }
        try:
            requests.post(webhook, json=payload, timeout=5).raise_for_status()
        except Exception as e:
            self.logger.error("Slack alert failed", extra={"error": str(e)})

    def _send_email(self, alert: Alert) -> None:
        if os.getenv("ALERT_EMAIL_ENABLED", "false").lower() != "true":
            return
        host = os.getenv("SMTP_SERVER", "")
        port = int(os.getenv("SMTP_PORT", "587"))
        user = os.getenv("SMTP_USERNAME", "")
        pwd = os.getenv("SMTP_PASSWORD", "")
        recipients = [r for r in os.getenv("ALERT_EMAIL_RECIPIENTS", "").split(",") if r]
        if not host or not user or not pwd or not recipients:
            return
        msg = MIMEMultipart()
        msg["From"] = user
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = f"[{alert.severity.value.upper()}] CICOP: {alert.title}"
        body = f"Component: {alert.component}\nMessage: {alert.message}\nMetadata: {json.dumps(alert.metadata, indent=2)}"
        msg.attach(MIMEText(body, "plain"))
        try:
            server = smtplib.SMTP(host, port)
            server.starttls()
            server.login(user, pwd)
            server.send_message(msg)
            server.quit()
        except Exception as e:
            self.logger.error("Email alert failed", extra={"error": str(e)})


class HealthMonitor:
    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.logger = setup_logger("HealthMonitor")
        self.health_checks: Dict[str, Callable[[], bool]] = {}
        self.interval = int(os.getenv("HEALTH_INTERVAL_SECONDS", "30"))
        self.max_failures = int(os.getenv("HEALTH_MAX_FAILURES", "3"))
        self.fail_counts: Dict[str, int] = {}
        self._running = False

    def register(self, name: str, check: Callable[[], bool]) -> None:
        self.health_checks[name] = check
        self.fail_counts[name] = 0
        self.logger.info("Registered health check", extra={"name": name})

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        import threading

        threading.Thread(target=self._loop, daemon=True).start()
        self.logger.info("Health monitor started")

    def stop(self) -> None:
        self._running = False
        self.logger.info("Health monitor stopped")

    def _loop(self) -> None:
        while self._running:
            for name, check in self.health_checks.items():
                ok = False
                try:
                    ok = bool(check())
                except Exception as e:
                    self.logger.error("Health check threw", extra={"name": name, "error": str(e)})
                if ok:
                    if self.fail_counts[name] > 0:
                        self.alert_manager.send_alert(
                            Alert(AlertSeverity.LOW, f"{name} recovered", "Service healthy", name)
                        )
                    self.fail_counts[name] = 0
                else:
                    self.fail_counts[name] += 1
                    if self.fail_counts[name] >= self.max_failures:
                        self.alert_manager.send_alert(
                            Alert(AlertSeverity.CRITICAL, f"{name} down", "Health check failing", name, metadata={"failures": self.fail_counts[name]})
                        )
                    elif self.fail_counts[name] == 1:
                        self.alert_manager.send_alert(
                            Alert(AlertSeverity.HIGH, f"{name} unhealthy", "Health check failed", name)
                        )
            time.sleep(self.interval)


# Convenience health checks

def api_health_check(url: str = "http://localhost:8000/health") -> bool:
    try:
        r = requests.get(url, timeout=3)
        return r.ok
    except Exception:
        return False


def setup_monitoring() -> tuple[AlertManager, HealthMonitor]:
    am = AlertManager({})
    hm = HealthMonitor(am)
    hm.register("api", api_health_check)
    return am, hm
