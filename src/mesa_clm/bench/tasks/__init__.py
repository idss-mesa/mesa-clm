"""Bench tasks (mesa-anyjev ``bench/tasks``). Importing a task module registers it; the neon
tasks need a label store and are built by :func:`mesa_clm.bench.tasks.neon.tasks_from_store`."""

from mesa_clm.bench.tasks.base import (
    ALL_TIERS,
    MIN_HELDOUT_PER_CLASS,
    MIN_TRAIN_PER_CLASS,
    TASKS,
    Fold,
    Item,
    Task,
    fold_guard,
    get_task,
    register,
)

__all__ = [
    "ALL_TIERS",
    "MIN_HELDOUT_PER_CLASS",
    "MIN_TRAIN_PER_CLASS",
    "TASKS",
    "Fold",
    "Item",
    "Task",
    "fold_guard",
    "get_task",
    "register",
]
