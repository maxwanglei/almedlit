import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

EventHandler = Callable[[dict[str, Any]], None]
logger = logging.getLogger(__name__)


def _handler_name(handler: EventHandler) -> str:
    return getattr(handler, "__qualname__", repr(handler))


class EventBus:
    """Small in-process event bus.

    This mimics the planned internal event-driven loose coupling. When the lab
    profile (Celery workers + Redis) is enabled, handlers can be migrated to
    Celery signals or a durable event table without changing the publish API.
    """

    def __init__(self) -> None:
        self.handlers: dict[str, list[EventHandler]] = defaultdict(list)

    def subscribe(self, event_name: str, handler: EventHandler) -> None:
        if handler not in self.handlers[event_name]:
            self.handlers[event_name].append(handler)

    def subscriber_count(self, event_name: str) -> int:
        return len(self.handlers[event_name])

    def publish(self, event_name: str, payload: dict[str, Any]) -> None:
        for handler in self.handlers[event_name]:
            try:
                handler(payload)
            except Exception:
                logger.exception(
                    "Event handler %s failed while handling %s",
                    _handler_name(handler),
                    event_name,
                )


event_bus = EventBus()
