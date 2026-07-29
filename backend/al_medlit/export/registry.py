from dataclasses import dataclass, field
from typing import Protocol

from sqlalchemy.orm import Session

from al_medlit.core.exceptions import ValidationError
from al_medlit.lineage.models import AnnotationSet, CorpusSnapshot


@dataclass(frozen=True, slots=True)
class ExportContext:
    db: Session
    project_id: int
    corpus_snapshot: CorpusSnapshot
    annotation_set: AnnotationSet | None
    options: dict = field(default_factory=dict)


class ExportFormatPlugin(Protocol):
    key: str
    content_type: str
    extension: str

    def iter_rows(self, context: ExportContext): ...


class ExportFormatRegistry:
    def __init__(self) -> None:
        self._formats: dict[str, ExportFormatPlugin] = {}

    def register(self, plugin: ExportFormatPlugin, *, replace: bool = False) -> None:
        if not plugin.key:
            raise ValidationError("Export format key cannot be empty")
        if plugin.key in self._formats and not replace:
            raise ValidationError(f"Export format '{plugin.key}' is already registered")
        self._formats[plugin.key] = plugin

    def get(self, key: str) -> ExportFormatPlugin:
        try:
            return self._formats[key]
        except KeyError as exc:
            raise ValidationError(f"Unknown export format '{key}'") from exc

    def list(self) -> tuple[ExportFormatPlugin, ...]:
        return tuple(self._formats[key] for key in sorted(self._formats))


export_formats = ExportFormatRegistry()
