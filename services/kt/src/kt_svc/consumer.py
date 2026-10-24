"""Interaction-stream consumer (FR-3b): responses in, mastery out.
Why a Redis consumer group rather than a direct call from the assessment
service: a learner's response must be captured even when the KT updater is
slow, restarting or broken. The assessment service publishes and moves on; this
consumer reads at its own pace, and a failure here delays a mastery refresh
rather than losing a learner's answer or blocking their next question.
Idempotency is explicit. At-least-once delivery means this consumer WILL see
the same interaction twice after a restart, and a mastery update applied twice
is a learner credited with answering twice.
"""
from __future__ import annotations
import json
from loguru import logger

STREAM = "interactions"
GROUP = "kt-updater"
BLOCK_MS = 2000
BATCH = 64

def ensure_group(redis, stream: str = STREAM, group: str = GROUP) -> None:
    try:
        redis.xgroup_create(stream, group, id="0", mkstream=True)
    except Exception as exc: # noqa: BLE001 - BUSYGROUP means it already exists
        if "BUSYGROUP" not in str(exc):
            raise

def parse(message: dict) -> dict:
    payload = message.get("payload")
    data = json.loads(payload) if isinstance(payload, (str, bytes)) else message
    required = ("interaction_id", "learner_id", "item_id", "skill_ids", "correct")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"interaction event missing {missing}")
    return data

def already_applied(conn, interaction_id, sa) -> bool:
    """The guard against double-counting. A tiny table, checked before every
    update, is far cheaper than reconciling a mastery map that drifted upward
    every time a pod restarted."""
    return conn.execute(sa.text(
        "SELECT 1 FROM mastery_applied WHERE interaction_id = :i"),
        {"i": interaction_id}).first() is not None

def mark_applied(conn, interaction_id, sa) -> None:
    conn.execute(sa.text(
        "INSERT INTO mastery_applied (interaction_id, applied_at) "
        "VALUES (:i, now()) ON CONFLICT DO NOTHING"), {"i": interaction_id})

def run(redis, engine, updater, stream: str = STREAM, group: str = GROUP,
        consumer: str = "kt-1", max_batches: int | None = None) -> dict:
    """The loop. Acknowledges only AFTER the update commits: acknowledging first
    means a crash between the two loses the update silently, and the whole
    point of the stream is that nothing is lost."""
    import sqlalchemy as sa
    ensure_group(redis, stream, group)
    stats = {"read": 0, "applied": 0, "duplicates": 0, "failed": 0}
    batches = 0
    while max_batches is None or batches < max_batches:
        batches += 1
        messages = redis.xreadgroup(group, consumer, {stream: ">"},
                                    count=BATCH, block=BLOCK_MS)
        if not messages:
            if max_batches is not None:
                break
            continue
        for _, entries in messages:
            for message_id, fields in entries:
                stats["read"] += 1
                try:
                    event = parse(fields)
                    with engine.begin() as conn:
                        if already_applied(conn, event["interaction_id"], sa):
                            stats["duplicates"] += 1
                        else:
                            updater.apply(conn, event)
                            mark_applied(conn, event["interaction_id"], sa)
                            stats["applied"] += 1
                    redis.xack(stream, group, message_id)
                except Exception as exc: # noqa: BLE001
                    # NOT acknowledged: it stays in the pending list for the
                    # claim-and-retry job rather than vanishing
                    stats["failed"] += 1
                    logger.error("interaction {} failed: {}", message_id, exc)
    return stats

