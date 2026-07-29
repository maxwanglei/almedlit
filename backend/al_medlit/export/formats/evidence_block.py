from collections import defaultdict

from al_medlit.corpus.models import Document, DocumentSentence
from al_medlit.evidence.models import EvidenceTargetVersion
from al_medlit.export.registry import ExportContext
from al_medlit.training.windowing import (
    EvidenceBlockWindowBuilder,
    GoldBlockRange,
    ReviewedInterval,
    TargetCondition,
    WindowBuilderConfig,
    WindowSentenceInput,
)


class EvidenceBlocksJSONLFormat:
    key = "evidence-blocks-v1"
    content_type = "application/x-ndjson"
    extension = "jsonl"

    def iter_rows(self, context: ExportContext):
        annotation_set = context.annotation_set
        if annotation_set is None:
            raise ValueError("evidence-blocks-v1 requires an annotation set")
        for item in annotation_set.items:
            document = context.db.get(Document, item.document_id)
            target_version = context.db.get(EvidenceTargetVersion, item.target_version_id)
            target = target_version.target
            yield {
                "schema_version": "evidence-blocks-v1",
                "document_id": item.document_id,
                "source_document_id": document.external_id,
                "target": {
                    "id": target_version.id,
                    "key": target.key,
                    "name": target.name,
                    "text": target_version.text,
                },
                "block": {
                    "start_sentence_ordinal": item.start_sentence_ordinal,
                    "end_sentence_ordinal": item.end_sentence_ordinal,
                    "text": item.block_text,
                    "section_paths": item.section_paths,
                    "char_start": item.start_char,
                    "char_end": item.end_char,
                    "labels": item.labels,
                },
                "provenance": {
                    "structure_version_id": item.structure_version_id,
                    "annotation_id": item.source_annotation_id,
                    "annotation_set_id": annotation_set.id,
                    "corpus_snapshot_id": annotation_set.corpus_snapshot_id,
                    "guideline_version_id": item.guideline_version_id,
                    "source": item.source,
                },
            }


class TrainingWindowsJSONLFormat:
    key = "training-windows-v1"
    content_type = "application/x-ndjson"
    extension = "jsonl"

    def iter_rows(self, context: ExportContext):
        annotation_set = context.annotation_set
        if annotation_set is None:
            raise ValueError("training-windows-v1 requires an annotation set")
        options = context.options
        config = WindowBuilderConfig(
            max_tokens=int(options.get("max_tokens", 4096)),
            overlap_tokens=int(options.get("overlap_tokens", 512)),
            reserved_special_tokens=int(options.get("reserved_special_tokens", 4)),
            target_conditioning=bool(options.get("target_conditioning", True)),
        )
        # Export-time counts are deterministic estimates. The trainer invokes the
        # same builder with its exact pinned tokenizer before launching a job.
        builder = EvidenceBlockWindowBuilder(_whitespace_token_count, config)
        blocks_by_scope = defaultdict(list)
        for item in annotation_set.items:
            blocks_by_scope[(item.document_id, item.target_version_id)].append(item)
        reviews_by_scope = defaultdict(list)
        for region in annotation_set.reviewed_regions:
            reviews_by_scope[(region.document_id, region.target_version_id)].append(region)

        for snapshot_document in annotation_set.corpus_snapshot.documents:
            document = context.db.get(Document, snapshot_document.document_id)
            sentences = (
                context.db.query(DocumentSentence)
                .filter(
                    DocumentSentence.structure_version_id
                    == snapshot_document.structure_version_id
                )
                .order_by(DocumentSentence.ordinal)
                .all()
            )
            sentence_inputs = [
                WindowSentenceInput(
                    id=sentence.id,
                    ordinal=sentence.ordinal,
                    paragraph_ordinal=sentence.paragraph.ordinal,
                    section_path=tuple(sentence.section.path or []),
                    text=document.text[sentence.start_offset : sentence.end_offset],
                    start_char=sentence.start_offset,
                    end_char=sentence.end_offset,
                )
                for sentence in sentences
            ]
            for target_version_id in annotation_set.target_version_ids:
                reviewed_regions = reviews_by_scope[(document.id, target_version_id)]
                if not reviewed_regions:
                    # A corpus snapshot may contain documents that were never
                    # assigned for every target. Do not emit all-IGNORE training
                    # windows for those unreviewed scopes.
                    continue
                target_version = context.db.get(EvidenceTargetVersion, target_version_id)
                if target_version.target.project_id != context.project_id:
                    continue
                result = builder.build(
                    document_id=document.id,
                    structure_version_id=snapshot_document.structure_version_id,
                    target=TargetCondition(
                        id=target_version.id,
                        key=target_version.target.key,
                        name=target_version.target.name,
                        text=target_version.text,
                    ),
                    sentences=sentence_inputs,
                    gold_blocks=[
                        GoldBlockRange(
                            id=item.id,
                            start_ordinal=item.start_sentence_ordinal,
                            end_ordinal=item.end_sentence_ordinal,
                        )
                        for item in blocks_by_scope[(document.id, target_version_id)]
                    ],
                    reviewed_intervals=[
                        ReviewedInterval(
                            start_ordinal=region.start_sentence_ordinal,
                            end_ordinal=region.end_sentence_ordinal,
                        )
                        for region in reviewed_regions
                    ],
                )
                for window in result.windows:
                    if all(sentence.label == "IGNORE" for sentence in window.sentences):
                        continue
                    yield {
                        "schema_version": "training-windows-v1",
                        "document_id": document.id,
                        "structure_version_id": snapshot_document.structure_version_id,
                        "target": {
                            "id": target_version.id,
                            "key": target_version.target.key,
                            "name": target_version.target.name,
                            "text": target_version.text,
                        },
                        "split": snapshot_document.split,
                        "window": {
                            "id": window.id,
                            "start_sentence_ordinal": window.start_sentence_ordinal,
                            "end_sentence_ordinal": window.end_sentence_ordinal,
                            "token_count_estimate": window.token_count,
                        },
                        "sentences": [
                            {
                                "id": sentence.id,
                                "ordinal": sentence.ordinal,
                                "paragraph_ordinal": sentence.paragraph_ordinal,
                                "section_path": list(sentence.section_path),
                                "text": sentence.text,
                                "label": sentence.label,
                                "reviewed": sentence.reviewed,
                            }
                            for sentence in window.sentences
                        ],
                        "provenance": {
                            "corpus_snapshot_id": annotation_set.corpus_snapshot_id,
                            "annotation_set_id": annotation_set.id,
                            "guideline_version_ids": annotation_set.guideline_version_ids,
                            "window_builder": "evidence-block-window-builder-v1",
                            "token_counting": "whitespace-estimate-v1",
                        },
                    }


def _whitespace_token_count(text: str) -> int:
    return max(1, len(text.split()))
