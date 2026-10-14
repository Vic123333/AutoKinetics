"""Open chemical data sources: PubChem (identity), NIST WebBook (thermochemistry),
bundled NASA thermo database (offline)."""

from .lookup import SpeciesInfo, lookup_species, thermo_candidates

__all__ = ["SpeciesInfo", "lookup_species", "thermo_candidates"]
