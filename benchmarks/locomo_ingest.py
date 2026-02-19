#!/usr/bin/env python3
"""
LoCoMo ingestion script - hardened version with automatic embedding generation.

Usage:
    python -m benchmarks.locomo_ingest --conversation 0 --fresh
    python -m benchmarks.locomo_ingest --all --fresh
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add rage-substrate to path
RAGE_SUBSTRATE_PATH = Path(__file__).parent.parent.parent / "rage-substrate"
if RAGE_SUBSTRATE_PATH.exists():
    sys.path.insert(0, str(RAGE_SUBSTRATE_PATH))

from benchmarks.locomo_connector import LoCoMoConnector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest LoCoMo conversations to RAGE")
    parser.add_argument(
        "--data", 
        type=str, 
        default="data/locomo/locomo10.json",
        help="Path to locomo10.json"
    )
    parser.add_argument(
        "--db", 
        type=str, 
        default="data/locomo/locomo_substrate.db",
        help="Path to RAGE database"
    )
    parser.add_argument(
        "--conversation", 
        type=int, 
        default=None,
        help="Index of specific conversation to ingest (0-9)"
    )
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Ingest all conversations"
    )
    parser.add_argument(
        "--fresh", 
        action="store_true",
        help="Clear database before ingesting"
    )
    parser.add_argument(
        "--skip-embeddings", 
        action="store_true",
        help="Skip embedding generation (faster but find/context won't work)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    if args.conversation is None and not args.all:
        parser.error("Specify --conversation N or --all")
    
    data_path = Path(args.data)
    db_path = Path(args.db)
    
    # Ensure data directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clear database if --fresh
    if args.fresh and db_path.exists():
        print(f"🗑️  Removing existing database: {db_path}")
        db_path.unlink()
    
    print("=" * 60)
    print("LoCoMo Ingestion")
    print("=" * 60)
    print(f"Data: {data_path}")
    print(f"Database: {db_path}")
    print()
    
    start_time = datetime.now()
    
    try:
        connector = LoCoMoConnector(data_path, db_path)
        
        if args.all:
            # Ingest all conversations
            stats = connector.ingest_all(verbose=args.verbose)
            print()
            print(f"✓ Ingested {stats['conversations']} conversations")
            print(f"  Sessions: {stats['sessions']}")
            print(f"  Messages: {stats['messages']}")
            print(f"  Frames: {stats['frames']}")
        else:
            # Ingest single conversation
            import json
            with open(data_path) as f:
                data = json.load(f)
            
            if args.conversation < 0 or args.conversation >= len(data):
                parser.error(f"Conversation index must be 0-{len(data)-1}")
            
            conv = data[args.conversation]
            sample_id = conv.get("sample_id", f"conv_{args.conversation}")
            
            print(f"Ingesting conversation {args.conversation}: {sample_id}")
            print()
            
            stats = connector.ingest_conversation(conv, verbose=True)
            
            print()
            print(f"✓ Ingested {sample_id}")
            print(f"  Sessions: {stats['sessions']}")
            print(f"  Messages: {stats['messages']}")
            print(f"  Frames: {stats['frames']}")
        
        # Generate embeddings unless skipped
        if not args.skip_embeddings:
            print()
            print("Generating embeddings...")
            substrate = connector._get_substrate()
            
            # Count frames needing embeddings
            import sqlite3
            conn = sqlite3.connect(db_path)
            total = conn.execute("SELECT COUNT(*) FROM frames WHERE embedding IS NULL").fetchone()[0]
            conn.close()
            
            if total > 0:
                print(f"  {total} frames need embeddings")
                embedded = substrate.embed_all_frames()
                print(f"  ✓ Generated {embedded} embeddings")
            else:
                print("  ✓ All frames already have embeddings")
        
        connector.close()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        print()
        print("=" * 60)
        print(f"✓ Complete in {elapsed:.1f}s")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        log.exception("Ingestion failed")
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
