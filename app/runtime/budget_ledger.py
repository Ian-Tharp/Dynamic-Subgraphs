"""Shared, atomic budget reservation ledger for concurrent spawn_subgraph siblings.

LangGraph merges a node's state writes only at the *end* of a superstep, so two
`spawn_subgraph` siblings scheduled in the same superstep both read the same
pre-merge `counters` snapshot — each would otherwise compute the full remaining
budget and over-allocate its child (the documented concurrent-sibling TOCTOU).

This ledger is a *live object* shared by reference through `state["metadata"]`
(the metadata reducer is a shallow merge, so the instance survives across
supersteps). LangGraph runs a superstep's nodes on a thread pool, so two siblings
may reserve concurrently; the lock makes `reserve`/`refund` atomic, so their
grants always sum to within the remaining budget. Allocation is first-come-greedy:
under contention the first sibling reserves the remaining and a later one can get
nothing and fail closed — safe (never overspending), with fair per-call sharing
left as a future refinement.

It does **not** replace the additive `counters` ledger — that still records
actual spend and rolls child spend up to the parent. This is the
*concurrent-allocation enforcement* layered on top: `reserve()` hands out at most
the remaining budget across all siblings, and `refund()` returns whatever a child
didn't spend so a later sibling can use it.

Reconciliation with `counters`: a child runs synchronously, so its spend lands in
`counters` only at the next superstep boundary. `reserve()` therefore tracks a
per-resource *pending* tally (reservations not yet visible in `counters`) keyed to
a *baseline* counters value; when `counters` advances (a new superstep absorbed the
prior pending spend) the pending tally resets, so a reservation is never counted
twice.
"""

from __future__ import annotations

import threading


class BudgetLedger:
    """Tracks in-flight spawn reservations so concurrent siblings can't overspend.

    Stateless about the budget itself — the caller passes the granted budget and
    the consumed-so-far counters on each `reserve()`; the ledger only owns the
    per-resource pending tally and its counters baseline.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[str, int] = {}
        self._baseline: dict[str, int] = {}

    def reserve(
        self,
        *,
        budgets: dict[str, int | None],
        consumed: dict[str, int],
    ) -> dict[str, int | None]:
        """Reserve the remaining allowance for one child; return its ceiling per
        resource (``None`` == unbounded for that resource).

        The sum of grants across siblings in one superstep can never exceed
        ``budget - consumed`` for any resource, because each grant decrements a
        shared pending tally that the next sibling reads.
        """
        with self._lock:
            grants: dict[str, int | None] = {}
            for key, budget in budgets.items():
                used = consumed.get(key, 0)
                # A new superstep advanced counters past our baseline, which means
                # counters already absorbed the prior pending spend — reset it.
                if used > self._baseline.get(key, 0):
                    self._pending[key] = 0
                    self._baseline[key] = used
                if budget is None:
                    grants[key] = None
                    continue
                available = max(0, budget - used - self._pending.get(key, 0))
                grants[key] = available
                self._pending[key] = self._pending.get(key, 0) + available
            return grants

    def refund(
        self,
        *,
        reserved: dict[str, int | None],
        spent: dict[str, int],
    ) -> None:
        """Return the unused portion (``reserved - spent``) of a reservation to the
        pool so an under-spending child frees budget for later siblings. Always
        safe to call (e.g. in a ``finally``); a child that never ran has
        ``spent == 0`` and refunds its whole reservation.
        """
        with self._lock:
            for key, ceiling in reserved.items():
                if ceiling is None:
                    continue
                used = spent.get(key, 0)
                unused = ceiling - used
                if unused > 0:
                    self._pending[key] = max(0, self._pending.get(key, 0) - unused)
