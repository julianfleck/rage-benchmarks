"""
LoCoMo connector - adapter for ingesting LoCoMo benchmark conversations to RAGE.

Modeled after OpenClaw connector pattern:
- Uses ExtractionSchema to configure extraction behavior
- Calls ingest_batch() for unified ingestion pipeline
- Windowed batch processing for efficiency

LoCoMo is a long-context conversation memory benchmark with multi-session
dialogues and QA pairs testing temporal reasoning, causal connections, and
long-term memory retention.

TODO (2026-02-24): Update to use new v2 ingestion pipeline
- Replace ingest_batch() with ingest_conversation() 
- New API: ingest_conversation(messages, source, session_id, db)
- messages format: [{"id": "...", "author": "...", "content": "...", "timestamp": "..."}]
- See rage_substrate/ingestion/pipeline.py for details
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add rage-substrate to path if needed
RAGE_SUBSTRATE_PATH = Path(__file__).parent.parent.parent / "rage-substrate"
if RAGE_SUBSTRATE_PATH.exists():
    sys.path.insert(0, str(RAGE_SUBSTRATE_PATH))

from rage_substrate.core.substrate import Substrate
from rage_substrate.ingestion.pipeline import ingest_batch
from rage_substrate.ingestion.schema import ExtractionSchema, ContainerSpec

log = logging.getLogger(__name__)


# =============================================================================
# LoCoMo Extraction Schema
# =============================================================================

LOCOMO_SCHEMA = ExtractionSchema(
    containers=[
        ContainerSpec(
            frame_type="message",
            title_template="{author}: {content[:40]}...",
            fields={
                "author": "item.author",
                "timestamp": "item.timestamp",
            },
        )
    ],
    hints="""
        - Long-term conversation memory benchmark
        - Track events, temporal relationships, causal connections
        - Preserve speaker personas and life events
        - Extract specific dates, times, and temporal references
        - Note relationships between speakers
        - Capture life events, decisions, and personal milestones
        - Mark emotional content and support interactions
        - Focus on facts that could be asked about later
    """,
    focus_types=["event", "fact", "relationship", "temporal"],
    min_content_length=20,
)


def parse_locomo_date(date_str: str) -> Optional[datetime]:
    """
    Parse LoCoMo date format like '1:56 pm on 8 May, 2023'.
    
    Returns datetime or None if parsing fails.
    """
    if not date_str:
        return None
    
    # Common formats in LoCoMo
    formats = [
        "%I:%M %p on %d %B, %Y",  # "1:56 pm on 8 May, 2023"
        "%I:%M %p on %B %d, %Y",  # "1:56 pm on May 8, 2023"
        "%d %B, %Y",              # "8 May, 2023"
        "%B %d, %Y",              # "May 8, 2023"
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # Fallback: try to extract year at least
    import re
    year_match = re.search(r'\d{4}', date_str)
    if year_match:
        try:
            return datetime(int(year_match.group()), 6, 15)
        except:
            pass
    
    return None


class LoCoMoConnector:
    """Connector for ingesting LoCoMo benchmark conversations into RAGE.
    
    Responsibilities:
    - Load and parse locomo10.json
    - Convert conversation turns to items for ingest_batch
    - Track ingestion progress
    
    Usage:
        connector = LoCoMoConnector(data_path, db_path)
        connector.ingest_all(limit=5)
    """
    
    def __init__(
        self,
        data_path: str | Path,
        db_path: str | Path,
        substrate: Optional[Substrate] = None,
    ):
        """Initialize connector.
        
        Args:
            data_path: Path to locomo10.json
            db_path: Path to RAGE substrate database
            substrate: Optional Substrate instance (created if not provided)
        """
        self.data_path = Path(data_path)
        self.db_path = Path(db_path)
        self._substrate = substrate
        self._data: Optional[List[Dict]] = None
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"LoCoMo data not found: {data_path}")
    
    def _get_substrate(self) -> Substrate:
        """Get or create the Substrate instance."""
        if self._substrate is None:
            self._substrate = Substrate("locomo_benchmark", db_path=str(self.db_path))
        return self._substrate
    
    def _load_data(self) -> List[Dict]:
        """Load LoCoMo data from JSON file."""
        if self._data is None:
            with open(self.data_path) as f:
                self._data = json.load(f)
        return self._data
    
    def _session_exists(self, sample_id: str, session_num: int) -> bool:
        """Check if a session has already been ingested.
        
        Args:
            sample_id: The LoCoMo sample ID
            session_num: The session number
            
        Returns:
            True if the session already has frames in the substrate
        """
        session_id = f"locomo_{sample_id}_s{session_num}"
        
        try:
            # Direct SQL check for frames with this session_id in slots
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            count = conn.execute(
                "SELECT COUNT(*) FROM frames WHERE json_extract(slots, '$.session_context') = ?",
                (session_id,)
            ).fetchone()[0]
            conn.close()
            return count > 0
        except Exception as e:
            log.warning(f"Error checking session existence for {session_id}: {e}")
            return False
    
    def _extract_sessions(self, conv_data: Dict) -> List[Dict]:
        """Extract sessions from a conversation.
        
        Returns list of session dicts with:
            - session_num: int
            - date_str: original date string
            - date_time: datetime or None
            - speaker_a: str
            - speaker_b: str
            - turns: list of turn dicts
        """
        speaker_a = conv_data.get("speaker_a", "Speaker A")
        speaker_b = conv_data.get("speaker_b", "Speaker B")
        
        sessions = []
        
        # Find all session keys
        session_nums = set()
        for key in conv_data.keys():
            if key.startswith("session_") and not key.endswith("_date_time"):
                try:
                    num = int(key.split("_")[1])
                    session_nums.add(num)
                except (IndexError, ValueError):
                    continue
        
        for num in sorted(session_nums):
            session_key = f"session_{num}"
            date_key = f"session_{num}_date_time"
            
            turns = conv_data.get(session_key, [])
            date_str = conv_data.get(date_key, "")
            
            if not turns:
                continue
            
            sessions.append({
                "session_num": num,
                "date_str": date_str,
                "date_time": parse_locomo_date(date_str),
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "turns": turns,
            })
        
        return sessions
    
    def ingest_conversation(
        self,
        conv_item: Dict[str, Any],
        verbose: bool = False,
    ) -> Dict[str, int]:
        """Ingest a single LoCoMo conversation.
        
        Args:
            conv_item: Full conversation item from locomo10.json
            verbose: Enable verbose output
            
        Returns:
            Stats dict with session_count and message_count
        """
        sample_id = conv_item.get("sample_id", "unknown")
        conv_data = conv_item.get("conversation", {})
        
        stats = {"sessions": 0, "messages": 0, "frames": 0}
        
        sessions = self._extract_sessions(conv_data)
        substrate = self._get_substrate()
        
        for session in sessions:
            session_num = session["session_num"]
            dt = session["date_time"]
            
            # Check if session already ingested (resume capability)
            if self._session_exists(sample_id, session_num):
                log.info(f"Session {session_num} already ingested, skipping")
                if verbose:
                    print(f"  Session {session_num} already ingested, skipping", flush=True)
                continue
            
            # Convert turns to items for ingest_batch
            items = []
            for turn in session["turns"]:
                speaker = turn.get("speaker", "Unknown")
                dia_id = turn.get("dia_id", "")
                text = turn.get("text", "")
                
                if not text.strip():
                    continue
                
                items.append({
                    "id": dia_id,
                    "content": text,
                    "author": speaker,
                    "timestamp": dt.isoformat() if dt else None,
                })
            
            if not items:
                continue
            
            # Ingest via unified pipeline with schema
            result = ingest_batch(
                items=items,
                schema=LOCOMO_SCHEMA,
                session_id=f"locomo_{sample_id}_s{session_num}",
                structural_territory=f"/locomo/{sample_id}",
                substrate=substrate,
            )
            
            if result.success:
                stats["sessions"] += 1
                stats["messages"] += result.item_count
                stats["frames"] += result.frame_count
                
                if verbose:
                    print(f"  Session {session_num}: {result.item_count} messages → {result.frame_count} frames", flush=True)
            else:
                log.error(f"Failed to ingest session {session_num}: {result.error}")
                if verbose:
                    print(f"  Session {session_num}: FAILED - {result.error}")
        
        return stats
    
    def ingest_all(
        self,
        limit: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict[str, int]:
        """Ingest all LoCoMo conversations.
        
        Args:
            limit: Optional limit on number of conversations
            verbose: Enable verbose output
            
        Returns:
            Aggregate stats dict
        """
        data = self._load_data()
        
        if limit:
            data = data[:limit]
        
        total_stats = {
            "conversations": 0,
            "sessions": 0,
            "messages": 0,
            "frames": 0,
        }
        
        print(f"Ingesting {len(data)} conversations...", flush=True)
        
        for i, conv_item in enumerate(data):
            sample_id = conv_item.get("sample_id", f"conv_{i}")
            
            if verbose:
                print(f"\n[{i+1}/{len(data)}] {sample_id}")
            
            stats = self.ingest_conversation(conv_item, verbose=verbose)
            
            total_stats["conversations"] += 1
            total_stats["sessions"] += stats["sessions"]
            total_stats["messages"] += stats["messages"]
            total_stats["frames"] += stats["frames"]
            
            if not verbose and (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(data)} conversations...")
        
        return total_stats
    
    def close(self):
        """Close substrate connection."""
        if self._substrate:
            self._substrate.close()
            self._substrate = None
