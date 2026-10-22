import logging
import re


class ActionTokenAccessFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple) and len(record.args) == 5:
            client, method, path, version, status = record.args
            if isinstance(path, str):
                path = re.sub(
                    r"^(/api/(?:invites|account-actions))/[^/?]+",
                    r"\1/[redacted]",
                    path,
                )
                record.args = (client, method, path, version, status)
        return True
