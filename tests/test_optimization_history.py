"""Tests for the resumable optimization history store."""
from src.optimization.history import HistoryStore, IterationRecord


def _rec(i, score, source="search", status="ok"):
    return IterationRecord(
        iteration=i, score=score, parameters={"generate.max_speed": 1.3 + i * 0.01},
        metrics={"timing_accuracy": score}, status=status, source=source,
    )


def test_append_and_read_back(tmp_path):
    store = HistoryStore(tmp_path)
    store.append(_rec(0, 0.5))
    store.append(_rec(1, 0.7))
    recs = store.records()
    assert [r.iteration for r in recs] == [0, 1]
    assert recs[1].score == 0.7


def test_best_tracks_highest_score(tmp_path):
    store = HistoryStore(tmp_path)
    store.append(_rec(0, 0.5))
    store.append(_rec(1, 0.9))
    store.append(_rec(2, 0.6))
    best = store.best()
    assert best.iteration == 1 and best.score == 0.9


def test_failed_records_do_not_become_best(tmp_path):
    store = HistoryStore(tmp_path)
    store.append(_rec(0, 0.5))
    store.append(IterationRecord(iteration=1, score=None, parameters={}, status="failed", error="boom"))
    assert store.best().iteration == 0
    assert store.next_iteration() == 2


def test_state_roundtrip_and_resume(tmp_path):
    store = HistoryStore(tmp_path)
    store.append(_rec(0, 0.5))
    store.save_state({"stale_steps": 3, "rng": [1, [2, 3], None]})
    resumed = store.resume()
    assert resumed["next_iteration"] == 1
    assert resumed["state"]["stale_steps"] == 3
    assert resumed["best"].score == 0.5
    assert resumed["n_records"] == 1


def test_summary_reports_progression(tmp_path):
    store = HistoryStore(tmp_path)
    for i, s in enumerate([0.4, 0.6, 0.55]):
        store.append(_rec(i, s))
    summ = store.summary()
    assert summ["n_iterations"] == 3
    assert summ["best_score"] == 0.6
    assert summ["score_progression"] == [0.4, 0.6, 0.55]
