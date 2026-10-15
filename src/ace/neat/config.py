"""NEAT hyperparameters (GRAPH_NEAT_DESIGN.md section 10.1).

Three pydantic models, :class:`SpeciationConfig`, :class:`MutationConfig` and
:class:`NeatConfig`, with exactly the fields and defaults of the contract.

Contract readings / additions (documented per the contract's ambiguity rule):

* Nested defaults use ``Field(default_factory=...)`` instead of a shared default instance;
  the observable default values are identical.
* All models use ``extra="forbid"`` so a typo in a TOML config file is an error rather
  than a silently ignored key.
* Validation (all raise ``pydantic.ValidationError``, a ``ValueError`` subclass):
  every probability / fraction lies in ``[0, 1]``; ``population_size >= 2``;
  ``1 <= speciation.target_species <= population_size``;
  ``1 <= initial_min_workers <= initial_max_workers <= max_workers``; ``node_cost >= 0``.
  A few further sanity bounds are applied (non-negative coefficients and counts,
  ``max_attempts >= 1``, ``initial_threshold > 0``, ``stagnation_generations >= 1``,
  ``0 <= threshold_step < 1``).
* ``threshold_step`` is a *relative* step: each generation the speciator multiplies the
  threshold by ``1 - threshold_step`` (too few species) or ``1 + threshold_step`` (too
  many). Default 0.15.
* TOML layout for :meth:`NeatConfig.from_toml`: top-level keys are ``NeatConfig`` fields,
  with ``[speciation]`` and ``[mutation]`` tables for the nested models. A file whose only
  top-level key is a ``[neat]`` table is also accepted (the table is used as the root).
  Writing TOML is out of scope; :meth:`NeatConfig.to_toml_dict` returns a plain,
  TOML-compatible dict (``history_dir`` as a string) that round-trips through
  ``NeatConfig.model_validate``.
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SpeciationConfig(BaseModel):
    """Speciation, fitness sharing and stagnation settings (section 10.4)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    target_species: int = Field(default=5, ge=1)  # K
    c_excess: float = Field(default=1.0, ge=0.0)
    c_disjoint: float = Field(default=1.0, ge=0.0)
    c_properties: float = Field(default=0.5, ge=0.0)
    initial_threshold: float = Field(default=3.0, gt=0.0)
    # Relative per-generation adjustment toward K: threshold *= (1 - step) with too few
    # species, *= (1 + step) with too many (floor 0.05).
    threshold_step: float = Field(default=0.15, ge=0.0, lt=1.0)
    stagnation_generations: int = Field(default=15, ge=1)
    elitism_per_species: int = Field(default=1, ge=0)
    survival_fraction: float = Field(default=0.4, ge=0.0, le=1.0)


class MutationConfig(BaseModel):
    """Mutation operator probabilities and retry budget (section 10.2)."""

    model_config = ConfigDict(extra="forbid")

    p_property: float = Field(default=0.8, ge=0.0, le=1.0)
    p_property_per_gene: float = Field(default=0.1, ge=0.0, le=1.0)
    p_add_connection: float = Field(default=0.3, ge=0.0, le=1.0)
    p_add_worker: float = Field(default=0.15, ge=0.0, le=1.0)
    p_insert_merger: float = Field(default=0.05, ge=0.0, le=1.0)
    p_remove_connection: float = Field(default=0.05, ge=0.0, le=1.0)
    p_remove_worker: float = Field(default=0.05, ge=0.0, le=1.0)
    p_toggle_enable: float = Field(default=0.02, ge=0.0, le=1.0)
    max_attempts: int = Field(default=20, ge=1)  # per operator before giving up


class NeatConfig(BaseModel):
    """Top-level NEAT run configuration (section 10.1)."""

    model_config = ConfigDict(extra="forbid")

    population_size: int = Field(default=20, ge=2)
    generations: int = Field(default=10, ge=0)
    node_cost: float = Field(default=0.02, ge=0.0)  # lambda: adjusted = raw - lambda * n
    crossover_rate: float = Field(default=0.75, ge=0.0, le=1.0)
    initial_min_workers: int = Field(default=1, ge=1)
    initial_max_workers: int = Field(default=5, ge=1)
    max_workers: int = Field(default=12, ge=1)
    seed: int = 0
    speciation: SpeciationConfig = Field(default_factory=SpeciationConfig)
    mutation: MutationConfig = Field(default_factory=MutationConfig)
    history_dir: Path = Path(".ace/neat")

    @model_validator(mode="after")
    def _check_cross_field(self) -> NeatConfig:
        if self.speciation.target_species > self.population_size:
            raise ValueError(
                f"speciation.target_species ({self.speciation.target_species}) must be "
                f"<= population_size ({self.population_size})"
            )
        if not (self.initial_min_workers <= self.initial_max_workers <= self.max_workers):
            raise ValueError(
                "worker bounds must satisfy 1 <= initial_min_workers <= "
                "initial_max_workers <= max_workers; got "
                f"{self.initial_min_workers}, {self.initial_max_workers}, {self.max_workers}"
            )
        return self

    @classmethod
    def from_toml(cls, path: Path | str) -> NeatConfig:
        """Load and validate a config from a TOML file (layout in the module docstring)."""
        with Path(path).open("rb") as fh:
            data: dict[str, Any] = tomllib.load(fh)
        if set(data) == {"neat"} and isinstance(data["neat"], dict):
            data = data["neat"]
        return cls.model_validate(data)

    @classmethod
    def load(cls, path: Path | str | None) -> NeatConfig:
        """Return :meth:`from_toml` of ``path``, or the defaults when ``path`` is None."""
        if path is None:
            return cls()
        return cls.from_toml(path)

    def to_toml_dict(self) -> dict[str, Any]:
        """Return a plain dict (JSON/TOML-compatible scalars, nested tables as dicts)."""
        return self.model_dump(mode="json")
