"""
Pipeline Module

Provides a robust pipeline engine to record, replay, and manage data transformation steps:
- Step model with operation name, parameters, timestamp, and label.
- Pipeline with add_step, undo/redo, apply, clear.
- JSON (de)serialization for save/load, with an external operation registry.

Author: CortexX Team
Version: 1.0.0
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import pandas as pd


@dataclass
class Step:
    """
    Represents a single pipeline step.
    """
    op: str
    params: Dict[str, Any]
    ts: str
    label: Optional[str] = None


class Pipeline:
    """
    A sequence of data transformation steps with undo/redo and (de)serialization.
    """

    def __init__(self, registry: Optional[Dict[str, Callable]] = None, version: str = "1.0") -> None:
        self.steps: List[Step] = []
        self._undone: List[Step] = []
        self.registry: Dict[str, Callable] = registry or {}
        self.version: str = version

    @property
    def has_steps(self) -> bool:
        """Return True if the pipeline contains at least one step."""
        return len(self.steps) > 0

    def add_step(self, op: str, params: Dict[str, Any], label: Optional[str] = None) -> None:
        """
        Add a step to the pipeline and clear the redo stack.
        """
        step = Step(op=op, params=params, ts=datetime.utcnow().isoformat(), label=label)
        self.steps.append(step)
        self._undone.clear()

    def undo(self) -> bool:
        """
        Undo the last step (if any) by moving it to the redo stack.

        Returns:
            bool: True if an undo was performed, False if no steps to undo.
        """
        if not self.steps:
            return False
        self._undone.append(self.steps.pop())
        return True

    def redo(self) -> bool:
        """
        Redo the last undone step (if any) by moving it back to the steps list.

        Returns:
            bool: True if a redo was performed, False if no steps to redo.
        """
        if not self._undone:
            return False
        self.steps.append(self._undone.pop())
        return True

    def clear(self) -> None:
        """
        Clear all steps and redo stack.
        """
        self.steps.clear()
        self._undone.clear()

    def apply(self, df: pd.DataFrame, on_error: str = "skip") -> pd.DataFrame:
        """
        Apply the pipeline to a DataFrame by replaying all steps.

        Args:
            df (pd.DataFrame): Input DataFrame to transform.
            on_error (str): Error handling mode; 'skip' ignores failing steps, 'raise' raises exceptions.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        result = df.copy()
        for step in self.steps:
            func = self.registry.get(step.op)
            if func is None:
                if on_error == "raise":
                    raise ValueError(f"Unknown pipeline operation: {step.op}")
                else:
                    continue
            try:
                result = func(result, **(step.params or {}))
            except Exception:
                if on_error == "raise":
                    raise
                continue
        return result

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the pipeline to a dictionary.

        Returns:
            Dict[str, Any]: Serialized pipeline data.
        """
        return {
            "version": self.version,
            "steps": [asdict(s) for s in self.steps],
            "meta": {"exported_at": datetime.utcnow().isoformat()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], registry: Optional[Dict[str, Callable]] = None) -> "Pipeline":
        """
        Deserialize a pipeline from a dictionary.

        Args:
            data (Dict[str, Any]): Serialized pipeline data.
            registry (Optional[Dict[str, Callable]]): Registry of available transformation functions.

        Returns:
            Pipeline: A Pipeline instance loaded from data.
        """
        version = str(data.get("version", "1.0"))
        steps_data = data.get("steps", [])
        pl = cls(registry=registry, version=version)
        for sd in steps_data:
            pl.steps.append(
                Step(
                    op=sd.get("op"),
                    params=sd.get("params", {}),
                    ts=sd.get("ts", datetime.utcnow().isoformat()),
                    label=sd.get("label"),
                )
            )
        pl._undone.clear()
        return pl
