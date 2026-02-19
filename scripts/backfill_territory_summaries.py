#!/usr/bin/env python3
"""Backfill missing territory summaries from member frames using LLM rollup."""

import sqlite3
import os
import sys
import httpx
from pathlib import Path

# Config
DB_PATH = Path(__file__).parent.parent / "data/locomo/locomo_substrate.db"
MODEL = os.environ.get("ROLLUP_MODEL", "moonshotai/kimi-k2.5")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

def get_member_summaries(conn: sqlite3.Connection, territory_id: str) -> list[dict]:
    """Get summaries of frames that are members of this territory."""
    c = conn.cursor()
    # Use frame_memberships to find member frames
    c.execute("""
        SELECT f.id, f.frame_type, f.title, f.summary 
        FROM frames f
        JOIN frame_memberships fm ON f.id = fm.frame_id
        WHERE fm.container_id = ? 
          AND fm.membership_type = 'territory'
          AND f.frame_type != 'territory'
          AND f.summary IS NOT NULL AND f.summary != ''
        ORDER BY f.created_at
        LIMIT 50
    """, (territory_id,))
    return [{"id": r[0], "type": r[1], "title": r[2], "summary": r[3]} for r in c.fetchall()]

def generate_rollup_summary(territory_id: str, children: list[dict]) -> str:
    """Generate a rolled-up summary from member summaries using LLM."""
    if not children:
        return None
    
    child_text = "\n\n".join([
        f"[{c['type']}] {c['title']}\n{c['summary']}" 
        for c in children
    ])
    
    prompt = f"""You are summarizing a knowledge territory. Based on the following member frames, write a concise summary (2-4 sentences) that captures the main themes and content of this territory.

Territory: {territory_id}

Member frames:
{child_text}

Write a summary that would help someone understand what information is contained in this territory. Be specific about people, topics, and key details mentioned. Do not use phrases like "This territory contains" - just describe the content directly."""

    response = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,  # Kimi uses separate reasoning tokens
            "temperature": 0.3,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

def main():
    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY not set")
        sys.exit(1)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("""
        SELECT id, title FROM frames 
        WHERE frame_type = 'territory' AND (summary IS NULL OR summary = '')
    """)
    territories = c.fetchall()
    
    print(f"Found {len(territories)} territories without summaries")
    print(f"Using model: {MODEL}\n")
    
    for territory_id, title in territories:
        print(f"Processing: {territory_id}")
        
        members = get_member_summaries(conn, territory_id)
        print(f"  Found {len(members)} member frames with summaries")
        
        if not members:
            print("  Skipping - no members found")
            continue
        
        try:
            summary = generate_rollup_summary(territory_id, members)
            print(f"  Generated: {summary[:100]}...")
            
            c.execute("UPDATE frames SET summary = ? WHERE id = ?", (summary, territory_id))
            conn.commit()
            print("  ✓ Updated")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    conn.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
