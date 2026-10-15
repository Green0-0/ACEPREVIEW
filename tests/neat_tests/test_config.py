"""Tests for ace.neat.config."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ace.neat.config import MutationConfig, NeatConfig, SpeciationConfig


def test_defaults_match_contract() -> None:
    cfg = NeatConfig()
    assert cfg.population_size == 20
    assert cfg.generations == 10
    assert cfg.node_cost == 0.02
    assert cfg.crossover_rate == 0.75
    assert (cfg.initial_min_workers, cfg.initial_max_workers, cfg.max_workers) == (1, 5, 12)
    assert cfg.seed == 0
    assert cfg.history_dir == Path(".ace/neat")
    sp = cfg.speciation
    assert sp == SpeciationConfig()
    assert (sp.enabled, sp.target_species, sp.c_excess, sp.c_disjoint, sp.c_properties) == (
        True, 5, 1.0, 1.0, 0.5,
    )
    assert (sp.initial_threshold, sp.threshold_step, sp.stagnation_generations) == (3.0, 0.15, 15)
    assert (sp.elitism_per_species, sp.survival_fraction) == (1, 0.4)
    mu = cfg.mutation
    assert mu == MutationConfig()
    assert (mu.p_property, mu.p_property_per_gene, mu.p_add_connection) == (0.8, 0.1, 0.3)
    assert (mu.p_add_worker, mu.p_insert_merger, mu.p_remove_connection) == (0.15, 0.05, 0.05)
    assert (mu.p_remove_worker, mu.p_toggle_enable, mu.max_attempts) == (0.05, 0.02, 20)


def test_nested_defaults_not_shared() -> None:
    a, b = NeatConfig(), NeatConfig()
    a.mutation.p_add_worker = 0.9
    assert b.mutation.p_add_worker == 0.15


@pytest.mark.parametrize("field", ["p_property", "p_add_connection", "p_toggle_enable"])
@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_mutation_probabilities_bounded(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        MutationConfig(**{field: value})


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_probability_endpoints_ok(value: float) -> None:
    MutationConfig(p_add_worker=value)
    SpeciationConfig(survival_fraction=value)
    NeatConfig(crossover_rate=value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"crossover_rate": 1.5},
        {"population_size": 1},
        {"node_cost": -0.1},
        {"initial_min_workers": 0},
        {"initial_min_workers": 4, "initial_max_workers": 3},
        {"initial_max_workers": 13},
        {"speciation": {"target_species": 0}},
        {"population_size": 4, "speciation": {"target_species": 5}},
        {"speciation": {"survival_fraction": 1.2}},
        {"speciation": {"threshold_step": -0.1}},
        {"speciation": {"threshold_step": 1.0}},  # relative step: (1 - step) must stay > 0
        {"unknown_field": 1},
    ],
)
def test_invalid_configs_rejected(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        NeatConfig.model_validate(kwargs)


def test_boundary_valid() -> None:
    cfg = NeatConfig(
        population_size=5,
        speciation=SpeciationConfig(target_species=5),
        initial_min_workers=3,
        initial_max_workers=3,
        max_workers=3,
        node_cost=0.0,
    )
    assert cfg.speciation.target_species == 5


def test_from_toml_and_load(tmp_path: Path) -> None:
    path = tmp_path / "neat.toml"
    path.write_text(
        """
population_size = 8
node_cost = 0.1
history_dir = "runs/x"

[speciation]
target_species = 3
enabled = false

[mutation]
p_add_worker = 0.5
""",
        encoding="utf-8",
    )
    cfg = NeatConfig.from_toml(path)
    assert cfg.population_size == 8
    assert cfg.node_cost == 0.1
    assert cfg.history_dir == Path("runs/x")
    assert cfg.speciation.target_species == 3
    assert cfg.speciation.enabled is False
    assert cfg.mutation.p_add_worker == 0.5
    assert cfg.mutation.p_property == 0.8
    assert NeatConfig.load(path) == cfg
    assert NeatConfig.load(None) == NeatConfig()


def test_from_toml_neat_table(tmp_path: Path) -> None:
    path = tmp_path / "neat.toml"
    path.write_text("[neat]\ngenerations = 3\n[neat.mutation]\nmax_attempts = 2\n", encoding="utf-8")
    cfg = NeatConfig.from_toml(path)
    assert cfg.generations == 3
    assert cfg.mutation.max_attempts == 2


def test_from_toml_invalid(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("population_size = 1\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        NeatConfig.from_toml(path)


def test_to_toml_dict_roundtrip() -> None:
    cfg = NeatConfig(seed=7, history_dir=Path("a/b"), mutation=MutationConfig(p_add_worker=0.3))
    data = cfg.to_toml_dict()
    assert isinstance(data, dict)
    assert data["history_dir"] == "a/b"
    assert isinstance(data["speciation"], dict)
    assert data["mutation"]["p_add_worker"] == 0.3
    assert NeatConfig.model_validate(data) == cfg
