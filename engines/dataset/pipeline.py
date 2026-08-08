"""
AFE — Dataset Processing Pipeline Engine

Executes end-to-end dataset curation:
Ingestion → Cleaning → Deduplication → Chunking → Quality Scoring → Cleaned Storage
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List

from core.database.engine import get_session
from core.database.models.dataset import Dataset
from core.database.models.task_job import JobStatus, TaskJob
from core.events.bus import event_bus
from core.storage.sdk import get_storage_backend
from engines.dataset.ingestion import DatasetIngestion
from engines.dataset.cleaner import DatasetCleaner, CleaningOptions
from engines.dataset.dedup import DatasetDeduplicator
from engines.dataset.chunking import DatasetChunker
from engines.dataset.quality import DatasetQualityScorer

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    clean: bool = True
    deduplicate: bool = True
    chunk: bool = False
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_quality_score: float = 0.3
    text_field: str = "text"
    cleaning_options: Dict[str, Any] = None


class DatasetPipeline:
    """Orchestrates the multi-stage transformation and curation of dataset files."""

    @classmethod
    def run_pipeline(cls, job_id: str, dataset_id: str, config_dict: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Background task entrypoint executed by task_queue.
        """
        cfg_dict = config_dict or {}
        cleaning_opts = CleaningOptions(**cfg_dict.get("cleaning_options", {})) if "cleaning_options" in cfg_dict else CleaningOptions()
        config = PipelineConfig(
            clean=cfg_dict.get("clean", True),
            deduplicate=cfg_dict.get("deduplicate", True),
            chunk=cfg_dict.get("chunk", False),
            chunk_size=cfg_dict.get("chunk_size", 512),
            chunk_overlap=cfg_dict.get("chunk_overlap", 64),
            min_quality_score=cfg_dict.get("min_quality_score", 0.3),
            text_field=cfg_dict.get("text_field", "text"),
        )

        storage = get_storage_backend()
        
        with get_session() as session:
            dataset = session.get(Dataset, dataset_id)
            if not dataset:
                raise ValueError(f"Dataset {dataset_id} not found.")
            
            raw_key = dataset.storage_key
            raw_path = storage.get_url(raw_key)

        # Processing state
        deduplicator = DatasetDeduplicator() if config.deduplicate else None
        processed_records: List[Dict[str, Any]] = []
        
        total_ingested = 0
        total_cleaned = 0
        total_deduped = 0
        total_chunked = 0
        quality_scores: List[float] = []

        # Read records
        for record in DatasetIngestion.read_records(raw_path, text_field=config.text_field):
            total_ingested += 1

            # 1. Clean
            if config.clean:
                record = DatasetCleaner.clean_record(record, text_field=config.text_field, options=cleaning_opts)
                if not record:
                    continue
                total_cleaned += 1

            # 2. Deduplicate
            text_content = record.get(config.text_field, "")
            if config.deduplicate and deduplicator:
                is_dup, _, _ = deduplicator.is_duplicate(text_content)
                if is_dup:
                    continue
                total_deduped += 1

            # 3. Quality score
            q_res = DatasetQualityScorer.score_text(text_content)
            q_score = q_res.get("score", 1.0)
            if q_score < config.min_quality_score:
                continue
            quality_scores.append(q_score)
            record["_quality_score"] = q_score

            # 4. Chunk
            if config.chunk:
                chunks = DatasetChunker.chunk_record(
                    record,
                    text_field=config.text_field,
                    chunk_size=config.chunk_size,
                    overlap=config.chunk_overlap,
                )
                total_chunked += len(chunks)
                processed_records.extend(chunks)
            else:
                processed_records.append(record)

        # Write processed output JSONL to Storage SDK
        output_filename = f"processed_{dataset_id}.jsonl"
        output_key = f"datasets/{dataset_id}/{output_filename}"
        output_lines = [json.dumps(r) for r in processed_records]
        output_bytes = "\n".join(output_lines).encode("utf-8")
        
        storage.write(output_key, output_bytes)

        avg_quality = round(sum(quality_scores) / max(1, len(quality_scores)), 3)
        sample_preview = processed_records[:5]

        # Update Dataset DB record
        with get_session() as session:
            dataset = session.get(Dataset, dataset_id)
            if dataset:
                dataset.processed_key = output_key
                dataset.status = "ready"
                dataset.row_count = len(processed_records)
                dataset.size_bytes = len(output_bytes)
                dataset.quality_score = avg_quality
                dataset.sample_preview_json = json.dumps(sample_preview, default=str)
                dataset.lineage_json = json.dumps({
                    "ingested": total_ingested,
                    "cleaned": total_cleaned,
                    "deduplicated": total_deduped,
                    "chunked": total_chunked,
                    "final_rows": len(processed_records),
                    "avg_quality": avg_quality,
                    "pipeline_config": asdict(config),
                })

        result_summary = {
            "dataset_id": dataset_id,
            "ingested_count": total_ingested,
            "final_rows": len(processed_records),
            "output_key": output_key,
            "avg_quality": avg_quality,
            "size_bytes": len(output_bytes),
        }
        
        event_bus.emit("dataset.processed", result_summary)
        return result_summary
