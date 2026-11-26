"""Bounded observability fields for local-explainer request logs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real


@dataclass
class LocalExplanationObservability:
    """Provider counters that are safe to copy into structured log records."""

    provider_latency: float | None = None
    inference_batch_count: int | None = None

    def capture(self, value: object) -> None:
        """Capture numeric provider counters without retaining exception details."""
        provider = getattr(value, "provider", None)
        source = provider if provider is not None else value
        latency = getattr(source, "provider_latency", None)
        if (
            isinstance(latency, Real)
            and not isinstance(latency, bool)
            and math.isfinite(float(latency))
        ):
            self.provider_latency = float(latency)

        batch_count = getattr(source, "inference_batch_count", None)
        if isinstance(batch_count, int) and not isinstance(batch_count, bool):
            self.inference_batch_count = batch_count
