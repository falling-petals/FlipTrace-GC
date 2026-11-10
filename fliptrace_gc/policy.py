"""Batch-view cache policies from the archived FlipTrace-GC comparisons."""

from collections import defaultdict, deque


def schedule_prefix_reuse(schedule, slots):
    """Plan admission from resident and just-computed views by earliest next use."""
    future = defaultdict(deque)
    for step, batch in enumerate(schedule):
        for key in batch:
            future[key].append(step)
    resident = set()
    plan = []
    for batch in schedule:
        full_hit = all(key in resident for key in batch)
        for key in batch:
            future[key].popleft()
        available = resident.union(batch)
        ranked = sorted((key for key in available if future[key]), key=lambda key: (future[key][0], key))
        keep = set(ranked[:slots])
        plan.append((full_hit, tuple(sorted(resident - keep)), tuple(sorted(keep - resident))))
        resident = keep
    return plan


def lru_prefix_reuse(recency, batch, slots):
    """Update recency in draw order and return only net changes for the batch."""
    before = set(recency)
    full_hit = all(key in before for key in batch)
    for key in batch:
        recency[key] = None
        recency.move_to_end(key)
        if len(recency) > slots:
            recency.popitem(last=False)
    after = set(recency)
    return full_hit, tuple(sorted(before - after)), tuple(sorted(after - before))
