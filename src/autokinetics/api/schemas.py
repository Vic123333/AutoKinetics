"""Request/response models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class MechanismPayload(BaseModel):
    """A mechanism either as Cantera-style document (dict) or as YAML/.kin text."""

    document: dict[str, Any] | None = None
    yaml: str | None = Field(default=None, max_length=8_000_000)
    kin: str | None = Field(default=None, max_length=8_000_000)
    phase: str | None = None

    @field_validator("yaml", "kin")
    @classmethod
    def _not_blank(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("leerer Text")
        return v


class SimulationRequest(BaseModel):
    mechanism: MechanismPayload
    reactor: dict[str, Any]
    simulation: dict[str, Any]
    concentration_unit: Literal["mol/L", "mol/m3", "mmol/L"] = "mol/L"
    include_heat_release: bool = False


class EquilibriumRequest(BaseModel):
    mechanism: MechanismPayload
    reactor: dict[str, Any]
    mode: Literal["TP", "TV", "HP", "UV"] = "TP"
    basis: Literal["auto", "elements", "reactions"] = "auto"
    concentration_unit: Literal["mol/L", "mol/m3", "mmol/L"] = "mol/L"


class SensitivityRequest(BaseModel):
    mechanism: MechanismPayload
    reactor: dict[str, Any]
    simulation: dict[str, Any]
    target: str
    reactions: list[int] | None = None
    max_reactions: int = Field(default=25, ge=1, le=400)
    perturbation: float = Field(default=0.05, gt=0, lt=1)


class RopRequest(BaseModel):
    mechanism: MechanismPayload
    reactor: dict[str, Any]
    simulation: dict[str, Any]
    species: str
    top: int = Field(default=10, ge=1, le=100)


class OrderFitRequest(BaseModel):
    t: list[float] = Field(min_length=4, max_length=100_000)
    c: list[float] = Field(min_length=4, max_length=100_000)
    concentration_unit: str = "mol/L"


class ArrheniusRequest(BaseModel):
    T: list[float] = Field(min_length=3, max_length=10_000)
    k: list[float] = Field(min_length=3, max_length=10_000)
    modified: bool = False


class DataSeries(BaseModel):
    species: str
    t: list[float] = Field(min_length=1, max_length=100_000)
    c: list[float] = Field(min_length=1, max_length=100_000)
    sigma: list[float] | None = None


class FitParameter(BaseModel):
    reaction: int = Field(ge=0)
    param: Literal["A", "Ea"]


class FitRequest(BaseModel):
    mechanism: MechanismPayload
    reactor: dict[str, Any]
    simulation: dict[str, Any]
    data: list[DataSeries] = Field(min_length=1)
    parameters: list[FitParameter] = Field(min_length=1, max_length=20)
    concentration_unit: Literal["mol/L", "mol/m3", "mmol/L"] = "mol/L"


class NumericsRequest(BaseModel):
    mechanism: MechanismPayload
    reactor: dict[str, Any]
    simulation: dict[str, Any]
    budget_s: float = Field(default=20.0, gt=0, le=300)
    target_rtol: float = Field(default=1e-4, gt=1e-12, lt=0.1)
    candidates: list[str] | None = None
