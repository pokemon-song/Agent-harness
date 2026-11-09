"""Structural backend contract; importing it does not import a simulator or NumPy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .profile import RobotProfile

if TYPE_CHECKING:
    from ..types import SensorFrame, ToolCall


@runtime_checkable
class RobotBackend(Protocol):
    """Synchronous measured-action interface, separate from task evaluation.

    ``execute`` validates and performs one robot action, returning measured execution
    feedback. Acceptance alone is not completion. Task declarations and perception
    tools remain harness concerns. Every implementation must expose its own profile.

    ``cancel`` reports its actual cancellation semantics. The Go2 simulation runs
    actions synchronously, so a same-thread cancel can only observe an idle backend
    and request a measured stop. It does not promise concurrent action interruption,
    a hardware emergency stop, or a controller running during model inference.
    """

    profile: RobotProfile

    def capture(self) -> SensorFrame: ...

    def execute(self, call: ToolCall) -> dict: ...

    def stop(self) -> dict: ...

    def cancel(self) -> dict: ...

    def close(self) -> None: ...


@runtime_checkable
class IndependentTaskEvaluator(Protocol):
    """Privileged acceptance used by episode reporting, never passed to the model.

    The runner archives ``criteria`` and calls ``evaluate`` independently of tool
    declarations. Implementations own their task truth and cumulative safety state;
    a robot action result cannot substitute for this evaluator.
    """

    def criteria(self, standoff_m: float = 1.0, tolerance_m: float = 0.18) -> dict: ...

    def evaluate(self, standoff_m: float = 1.0, tolerance_m: float = 0.18) -> dict: ...


@runtime_checkable
class SimulationEpisodeBackend(RobotBackend, Protocol):
    """Additional contract required by the current paused-simulation episode runner.

    The generic harness only requires ``RobotBackend``. The benchmark runner also
    needs a simulation clock, recorded physical trajectory and an independent
    evaluator. It must validate the paused-clock profile, complete timestamped
    RGB-D/localization observations and evaluator separately before calling an API.

    Deadlines are runner-supplied episode limits observed by the synchronous
    controller; trajectory entries describe measured physical state, including
    stopping drift. No hardware or asynchronous execution behavior is implied.

    Runtime Protocol checks establish member presence, not attribute value types,
    return schemas or trustworthy measurements. In particular callers must check
    ``evaluator`` against ``IndependentTaskEvaluator`` as well.
    """

    evaluator: IndependentTaskEvaluator
    trajectory: list[dict]
    deadline_monotonic: float | None
    sim_deadline_s: float | None
