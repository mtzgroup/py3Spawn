"""nextfms -- a target-state skeleton for a resilient, serial-reading AIMS/AIMC/SH driver.

Two tiers, one seam:
  * HARD CORE (executor.py, checkpoint in driver.py, state PRNG plumbing) --
    all concurrency + resilience; rarely edited; sealed.
  * PHYSICS   (physics.py, toy_backend.py, and the body of advance()) --
    pure, serial-reading, student-approachable.
The seam is ElectronicStructureBackend.compute_one (down) and
executor.evaluate (up).
"""
from .state import SimState, TBF
from .es_provider import ESRequest, ESResult, Coupling, ElectronicStructureBackend
from .executor import ThreadedExecutor
from .toy_backend import ToyAvoidedCrossing
from . import physics, driver

__all__ = ["SimState", "TBF", "ESRequest", "ESResult", "Coupling",
           "ElectronicStructureBackend", "ThreadedExecutor",
           "ToyAvoidedCrossing", "physics", "driver"]
