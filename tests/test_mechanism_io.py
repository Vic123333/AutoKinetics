import numpy as np
import pytest

from autokinetics.compiled import compile_mechanism
from autokinetics.errors import MechanismError
from autokinetics.io import dump_mechanism, load_mechanism
from autokinetics.io.kin_legacy import load_kin
from autokinetics.library import list_examples, load_example


def test_all_examples_load_and_validate():
    examples = list_examples()
    assert len(examples) >= 12
    for ex in examples:
        mech, meta, _ = load_example(ex.id)
        errors = [i for i in mech.validate() if i.level == "error"]
        assert not errors, (ex.id, errors)
        assert meta.preset.get("reactor") and meta.preset.get("simulation"), ex.id
        assert meta.sources, ex.id


def test_yaml_roundtrip_preserves_rates(gri30):
    from autokinetics._ext._core import KineticsModel

    text = dump_mechanism(gri30)
    again = load_mechanism(text)
    assert again.n_species == gri30.n_species and again.n_reactions == gri30.n_reactions
    k1, k2 = KineticsModel(compile_mechanism(gri30)), KineticsModel(compile_mechanism(again))
    rng = np.random.default_rng(3)
    for _ in range(5):
        T, P = rng.uniform(600, 2500), rng.uniform(1e4, 1e6)
        c = rng.random(gri30.n_species) * 5
        np.testing.assert_allclose(k1.production_rates(T, P, c), k2.production_rates(T, P, c), rtol=1e-10, atol=1e-12)


def test_validation_detects_element_imbalance():
    text = """
units: {length: m, quantity: mol, activation-energy: J/mol}
phases: [{name: g, thermo: ideal-gas, species: [H2, O2, H2O]}]
species:
- {name: H2, composition: {H: 2}}
- {name: O2, composition: {O: 2}}
- {name: H2O, composition: {H: 2, O: 1}}
reactions:
- {equation: H2 + O2 => H2O, rate-constant: {A: 1, b: 0, Ea: 0}}
"""
    mech = load_mechanism(text)
    msgs = [i.message for i in mech.validate() if i.level == "error"]
    assert any("Elementbilanz" in m for m in msgs)
    with pytest.raises(MechanismError):
        compile_mechanism(mech)


def test_reversible_without_thermo_is_rejected():
    text = """
units: {length: dm, quantity: mol, activation-energy: kJ/mol}
phases: [{name: l, thermo: liquid, species: [A, B]}]
species: [{name: A}, {name: B}]
reactions: [{equation: A <=> B, rate-constant: {A: 1, b: 0, Ea: 0}}]
"""
    mech = load_mechanism(text)
    assert any("Rückreaktion nicht bestimmbar" in i.message for i in mech.validate())


def test_yaml_booleans_do_not_eat_species_named_NO():
    text = """
units: {length: m, quantity: mol, activation-energy: J/mol}
phases: [{name: g, thermo: ideal-gas, species: [NO, N, O]}]
species:
- {name: NO, composition: {N: 1, O: 1}}
- {name: N, composition: {N: 1}}
- {name: O, composition: {O: 1}}
reactions: [{equation: N + O => NO, rate-constant: {A: 1, b: 0, Ea: 0}}]
"""
    assert load_mechanism(text).species_names == ["NO", "N", "O"]


def test_unsupported_reaction_type_is_reported():
    text = """
units: {length: m, quantity: mol, activation-energy: J/mol}
phases: [{name: g, thermo: ideal-gas, species: [A, B]}]
species: [{name: A}, {name: B}]
reactions: [{equation: A => B, type: Chebyshev, temperature-range: [300, 2000]}]
"""
    with pytest.raises(MechanismError, match="nicht unterstützt"):
        load_mechanism(text)


def test_legacy_kin_import():
    import json

    doc = {"species": [{"name": "A", "start_concentration": 1.0}, {"name": "B", "start_concentration": 0}],
           "groups": [], "arrows": [{"start_id": 0, "end_id": 1, "rate_constant": "k1", "arrow_type": "Forward",
                                     "arrhenius_A": 0.5, "activation_energy_Ea": 0, "reaction_order": "",
                                     "temperature_exponent_n": 0}]}
    mech = load_kin(json.dumps(doc))
    assert mech.reactions[0].equation == "A => B"
    assert mech.reactions[0].rate.A == pytest.approx(0.5)
    assert mech.metadata["reactor"]["composition"] == {"A": 1.0}
