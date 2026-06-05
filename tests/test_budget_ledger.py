"""BudgetLedger — concurrent-sibling reservation math (the TOCTOU fix, core).

Pure unit tests of reserve / refund / counters-baseline reconciliation. The
end-to-end two-sibling regression (through the real executor) lives in
test_subgraph.py.
"""

from __future__ import annotations

from app.runtime.budget_ledger import BudgetLedger


def test_first_reservation_gets_the_full_remaining() -> None:
    # Arrange / Act
    ledger = BudgetLedger()
    grant = ledger.reserve(budgets={"llm_calls": 8}, consumed={"llm_calls": 1})

    # Assert — 8 budget - 1 consumed - 0 pending.
    assert grant == {"llm_calls": 7}


def test_concurrent_siblings_cannot_jointly_exceed_remaining() -> None:
    # Arrange — two siblings in one superstep read the SAME consumed snapshot.
    ledger = BudgetLedger()
    consumed = {"llm_calls": 1}  # a seed already spent 1; remaining = 7

    # Act — A reserves its ceiling, spends only 4 of 7, refunds the unused 3.
    a = ledger.reserve(budgets={"llm_calls": 8}, consumed=consumed)
    ledger.refund(reserved=a, spent={"llm_calls": 4})
    b = ledger.reserve(budgets={"llm_calls": 8}, consumed=consumed)

    # Assert — B only sees what A actually left (8 - 1 - 4 = 3); joint ≤ 7.
    assert a == {"llm_calls": 7}
    assert b == {"llm_calls": 3}


def test_second_sibling_starves_when_first_takes_all() -> None:
    # Arrange / Act
    ledger = BudgetLedger()
    consumed = {"llm_calls": 0}
    a = ledger.reserve(budgets={"llm_calls": 2}, consumed=consumed)
    ledger.refund(reserved=a, spent={"llm_calls": 2})  # A used its whole ceiling
    b = ledger.reserve(budgets={"llm_calls": 2}, consumed=consumed)

    # Assert — nothing left for B (caller fails the spawn closed).
    assert a == {"llm_calls": 2}
    assert b == {"llm_calls": 0}


def test_baseline_resets_when_counters_advance_next_superstep() -> None:
    # Arrange — superstep 2: consumed=1, A reserves + spends 2 (pending tracks 2).
    ledger = BudgetLedger()
    a = ledger.reserve(budgets={"llm_calls": 8}, consumed={"llm_calls": 1})
    ledger.refund(reserved=a, spent={"llm_calls": 2})  # pending == 2

    # Act — superstep 3: counters absorbed the 2 -> consumed=3.
    c = ledger.reserve(budgets={"llm_calls": 8}, consumed={"llm_calls": 3})

    # Assert — pending was reset, so it's 8-3-0=5, NOT a double-counted 8-3-2=3.
    assert c == {"llm_calls": 5}


def test_none_budget_is_unbounded() -> None:
    # Arrange / Act
    ledger = BudgetLedger()
    grant = ledger.reserve(
        budgets={"llm_calls": None, "nodes": 4}, consumed={"nodes": 1}
    )

    # Assert
    assert grant["llm_calls"] is None
    assert grant["nodes"] == 3


def test_llm_and_nodes_are_tracked_independently() -> None:
    # Arrange / Act
    ledger = BudgetLedger()
    a = ledger.reserve(
        budgets={"llm_calls": 8, "nodes": 12}, consumed={"llm_calls": 0, "nodes": 0}
    )
    ledger.refund(reserved=a, spent={"llm_calls": 3, "nodes": 5})
    b = ledger.reserve(
        budgets={"llm_calls": 8, "nodes": 12}, consumed={"llm_calls": 0, "nodes": 0}
    )

    # Assert — each resource keeps its own pool.
    assert a == {"llm_calls": 8, "nodes": 12}
    assert b == {"llm_calls": 5, "nodes": 7}


def test_refund_of_a_failed_child_returns_the_whole_reservation() -> None:
    # Arrange — a child that never ran refunds with spent == 0.
    ledger = BudgetLedger()
    consumed = {"llm_calls": 0}
    a = ledger.reserve(budgets={"llm_calls": 5}, consumed=consumed)
    ledger.refund(reserved=a, spent={"llm_calls": 0})

    # Act — the next sibling should see the full budget again.
    b = ledger.reserve(budgets={"llm_calls": 5}, consumed=consumed)

    # Assert
    assert a == {"llm_calls": 5}
    assert b == {"llm_calls": 5}
