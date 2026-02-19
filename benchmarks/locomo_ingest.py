"""LoCoMo data ingestion: Convert LoCoMo conversations to RAGE frames."""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add rage-substrate to path
RAGE_SUBSTRATE_PATH = Path(__file__).parent.parent.parent / "rage-substrate"
if RAGE_SUBSTRATE_PATH.exists():
    sys.path.insert(0, str(RAGE_SUBSTRATE_PATH))

try:
    from rage_substrate.core.substrate import Substrate
except ImportError:
    print("Error: Could not import rage_substrate", file=sys.stderr)
    print("Make sure rage-substrate is installed or in ../rage-substrate", file=sys.stderr)
    sys.exit(1)


# LoCoMo category mapping
CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal", 
    3: "commonsense",
    4: "multi-hop",
    5: "adversarial"
}


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
            # Use middle of year as approximation
            return datetime(int(year_match.group()), 6, 15)
        except:
            pass
    
    return None


def load_locomo_data(data_path: Path = None) -> List[Dict[str, Any]]:
    """Load LoCoMo dataset from JSON file."""
    if data_path is None:
        data_path = Path(__file__).parent.parent / "data" / "locomo" / "locomo10.json"
    
    with open(data_path) as f:
        return json.load(f)


def extract_conversations(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract session conversations from a LoCoMo item.
    
    Returns list of sessions, each with:
        - session_id: str
        - date_time: datetime or None
        - date_str: original date string
        - speaker_a: str
        - speaker_b: str
        - turns: list of turn dicts
    """
    conv = item.get("conversation", {})
    speaker_a = conv.get("speaker_a", "Speaker A")
    speaker_b = conv.get("speaker_b", "Speaker B")
    
    sessions = []
    
    # Find all session keys
    session_nums = set()
    for key in conv.keys():
        if key.startswith("session_") and not key.endswith("_date_time"):
            try:
                num = int(key.split("_")[1])
                session_nums.add(num)
            except (IndexError, ValueError):
                continue
    
    for num in sorted(session_nums):
        session_key = f"session_{num}"
        date_key = f"session_{num}_date_time"
        
        turns = conv.get(session_key, [])
        date_str = conv.get(date_key, "")
        
        if not turns:
            continue
        
        sessions.append({
            "session_id": session_key,
            "session_num": num,
            "date_str": date_str,
            "date_time": parse_locomo_date(date_str),
            "speaker_a": speaker_a,
            "speaker_b": speaker_b,
            "turns": turns
        })
    
    return sessions


def ingest_to_rage(
    data: List[Dict[str, Any]],
    substrate: Substrate,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Ingest LoCoMo conversations into RAGE substrate.
    
    Creates:
    - Container frame for each conversation
    - Session frames as children of conversation
    - Message frames for each turn as children of session
    
    Returns stats dict.
    """
    stats = {
        "conversations": 0,
        "sessions": 0,
        "messages": 0,
        "qa_pairs": 0
    }
    
    for item_idx, item in enumerate(data):
        sample_id = item.get("sample_id", f"conv_{item_idx}")
        
        # Create conversation container
        conv_frame = substrate.tools.execute_sync("create", {
            "title": f"LoCoMo Conversation: {sample_id}",
            "content": f"Conversation from LoCoMo benchmark dataset (sample_id: {sample_id})",
            "type": "container",
            "metadata": {
                "locomo_sample_id": sample_id,
                "qa_count": len(item.get("qa", []))
            }
        })
        
        if not conv_frame.success:
            print(f"Failed to create conversation frame: {conv_frame.error}", file=sys.stderr)
            continue
        
        conv_fid = conv_frame.data.get("fid")
        stats["conversations"] += 1
        
        if verbose:
            print(f"Created conversation {sample_id} ({conv_fid})")
        
        # Extract and ingest sessions
        sessions = extract_conversations(item)
        
        for session in sessions:
            session_id = session["session_id"]
            date_str = session["date_str"]
            dt = session["date_time"]
            
            # Create session frame
            session_content = f"Session {session['session_num']} between {session['speaker_a']} and {session['speaker_b']}"
            if date_str:
                session_content += f" on {date_str}"
            
            session_frame = substrate.tools.execute_sync("create", {
                "title": f"{sample_id} - Session {session['session_num']}",
                "content": session_content,
                "type": "container",
                "parent": conv_fid,
                "timestamp": dt.isoformat() if dt else None,
                "metadata": {
                    "locomo_session": session_id,
                    "date_str": date_str,
                    "turn_count": len(session["turns"])
                }
            })
            
            if not session_frame.success:
                print(f"Failed to create session frame: {session_frame.error}", file=sys.stderr)
                continue
            
            session_fid = session_frame.data.get("fid")
            stats["sessions"] += 1
            
            # Create message frames for each turn
            for turn in session["turns"]:
                speaker = turn.get("speaker", "Unknown")
                dia_id = turn.get("dia_id", "")
                text = turn.get("text", "")
                
                msg_frame = substrate.tools.execute_sync("create", {
                    "title": f"[{dia_id}] {speaker}",
                    "content": text,
                    "type": "message",
                    "parent": session_fid,
                    "timestamp": dt.isoformat() if dt else None,
                    "metadata": {
                        "locomo_dia_id": dia_id,
                        "speaker": speaker
                    }
                })
                
                if msg_frame.success:
                    stats["messages"] += 1
                else:
                    print(f"Failed to create message frame: {msg_frame.error}", file=sys.stderr)
        
        # Count QA pairs
        stats["qa_pairs"] += len(item.get("qa", []))
    
    return stats


def main():
    """Ingest LoCoMo data into RAGE substrate."""
    parser = argparse.ArgumentParser(description="Ingest LoCoMo data into RAGE")
    parser.add_argument(
        "--data", "-d",
        type=str,
        default=None,
        help="Path to locomo10.json (default: data/locomo/locomo10.json)"
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to substrate database"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of conversations to ingest"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing database and start fresh"
    )
    
    args = parser.parse_args()
    
    # Load data
    data_path = Path(args.data) if args.data else None
    print("Loading LoCoMo data...")
    data = load_locomo_data(data_path)
    
    if args.limit:
        data = data[:args.limit]
    
    print(f"Loaded {len(data)} conversations")
    
    # Initialize substrate
    if args.db:
        db_path = args.db
    else:
        db_path = str(Path(__file__).parent.parent / "data" / "locomo" / "locomo_substrate.db")
    
    if args.fresh:
        import os
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"Removed existing database: {db_path}")
    
    print(f"Using database: {db_path}")
    substrate = Substrate("locomo_benchmark", db_path=db_path)
    
    # Ingest
    print("\nIngesting to RAGE substrate...")
    stats = ingest_to_rage(data, substrate, verbose=args.verbose)
    
    substrate.close()
    
    # Print summary
    print("\n" + "=" * 50)
    print("INGESTION COMPLETE")
    print("=" * 50)
    print(f"Conversations: {stats['conversations']}")
    print(f"Sessions:      {stats['sessions']}")
    print(f"Messages:      {stats['messages']}")
    print(f"QA pairs:      {stats['qa_pairs']}")
    print(f"\nDatabase: {db_path}")


if __name__ == "__main__":
    main()
