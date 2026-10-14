"""Exception hierarchy. Every user-facing error carries a German message."""


class AutoKineticsError(Exception):
    """Base class for all expected (user-input related) errors."""


class MechanismError(AutoKineticsError):
    """Invalid mechanism definition (species, reactions, thermo, units)."""


class UnitError(MechanismError):
    """Unknown or inconsistent unit."""


class SimulationError(AutoKineticsError):
    """Integration failed or reactor configuration is invalid."""


class ExternalDataError(AutoKineticsError):
    """External data source (PubChem, NIST) unreachable or returned no data."""
