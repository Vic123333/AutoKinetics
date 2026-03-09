"""
DRF serializers with strict input validation.
All limits are driven by Django settings (MAX_SPECIES, MAX_REACTIONS).
"""
from __future__ import annotations
import re
from django.conf import settings
from rest_framework import serializers

# Allowed species name characters: letters, digits, subscript/superscript
# Unicode, parentheses, brackets, +, -, *, ·, %, space  (≤ 50 chars)
_SPECIES_NAME_RE = re.compile(r'^[\w\s\+\-\*·\[\]\(\)\u2080-\u209C\u00B0-\u00FF%]{1,50}$')

# Absolute bounds that no physical system should exceed
MAX_CONCENTRATION = 1e6      # mol/L  (pure substance ~ 55 mol/L, far less in practice)
MAX_TEMPERATURE   = 20_000   # K      (plasma threshold)
MIN_TEMPERATURE   = 1        # K
MAX_T_END         = 1e9      # s
MAX_ARRHENIUS_A   = 1e50     # absolute upper bound
MIN_ARRHENIUS_A   = 0.0      # 0 means "no reaction at T=0, only at T>0"
MAX_EA            = 1_000_000  # J/mol  (>300 kJ/mol already very high barrier)
MAX_N             = 20.0       # temperature exponent upper bound (NIST max ~ 5)
MIN_N             = -10.0
MAX_STOICH        = 10.0


class StoichiometryPairField(serializers.ListField):
    """Validates [species_index, stoichiometric_coefficient] pairs."""
    child = serializers.ListField(min_length=2, max_length=2)

    def validate(self, value):
        value = super().validate(value)
        validated = []
        for pair in value:
            try:
                idx   = int(pair[0])
                stoich = float(pair[1])
            except (TypeError, ValueError) as exc:
                raise serializers.ValidationError(
                    f'Each entry must be [species_index, stoichiometry], got {pair!r}'
                ) from exc
            if idx < 0:
                raise serializers.ValidationError('Species index must be ≥ 0')
            if not (0 < stoich <= MAX_STOICH):
                raise serializers.ValidationError(
                    f'Stoichiometric coefficient must be in (0, {MAX_STOICH}]'
                )
            validated.append([idx, stoich])
        return validated


class SpeciesSerializer(serializers.Serializer):
    name                = serializers.CharField(max_length=50, trim_whitespace=True)
    start_concentration = serializers.FloatField(min_value=0.0, max_value=MAX_CONCENTRATION,
                                                  default=1.0)
    is_intermediate     = serializers.BooleanField(default=False)

    def validate_name(self, value: str) -> str:
        if not _SPECIES_NAME_RE.match(value):
            raise serializers.ValidationError(
                'Species name contains disallowed characters. '
                'Use letters, digits, and common chemistry notation.'
            )
        return value.strip()


class ReactionSerializer(serializers.Serializer):
    rate_label             = serializers.CharField(max_length=50, trim_whitespace=True)
    reactants              = StoichiometryPairField(min_length=1)
    products               = StoichiometryPairField(min_length=1)
    arrhenius_A            = serializers.FloatField(min_value=MIN_ARRHENIUS_A,
                                                     max_value=MAX_ARRHENIUS_A, default=1.0)
    activation_energy_Ea   = serializers.FloatField(min_value=0.0, max_value=MAX_EA, default=0.0)
    temperature_exponent_n = serializers.FloatField(min_value=MIN_N, max_value=MAX_N, default=0.0)
    reaction_order         = serializers.DictField(
        child=serializers.FloatField(min_value=0.0, max_value=10.0),
        required=False, default=dict
    )

    def validate_rate_label(self, value: str) -> str:
        if not re.match(r'^[\w\s\+\-\·₀-₉⁰-⁹]{1,50}$', value):
            raise serializers.ValidationError('Rate label contains disallowed characters.')
        return value.strip()

    def validate(self, data):
        # Verify stoichiometric pairs reference valid species (cross-check done in view)
        if not data.get('reactants') or not data.get('products'):
            raise serializers.ValidationError('Both reactants and products must be non-empty.')
        return data


class SimulationRequestSerializer(serializers.Serializer):
    species     = serializers.ListField(child=SpeciesSerializer(),   min_length=1)
    reactions   = serializers.ListField(child=ReactionSerializer(), min_length=1)
    temperature = serializers.FloatField(min_value=MIN_TEMPERATURE, max_value=MAX_TEMPERATURE,
                                          default=298.15)
    t_end       = serializers.FloatField(min_value=1e-6, max_value=MAX_T_END, default=100.0)
    n_points    = serializers.IntegerField(min_value=50, max_value=1000, default=400)

    def validate_species(self, value):
        max_sp = getattr(settings, 'MAX_SPECIES', 20)
        if len(value) > max_sp:
            raise serializers.ValidationError(
                f'Maximum {max_sp} species allowed per simulation.'
            )
        names = [s['name'] for s in value]
        if len(names) != len(set(names)):
            raise serializers.ValidationError('Species names must be unique.')
        return value

    def validate_reactions(self, value):
        max_rx = getattr(settings, 'MAX_REACTIONS', 50)
        if len(value) > max_rx:
            raise serializers.ValidationError(
                f'Maximum {max_rx} reactions allowed per simulation.'
            )
        return value

    def validate(self, data):
        n_species = len(data.get('species', []))
        for rxn in data.get('reactions', []):
            for pair_list in (rxn.get('reactants', []), rxn.get('products', [])):
                for idx, _ in pair_list:
                    if idx >= n_species:
                        raise serializers.ValidationError(
                            f'Reaction references species index {idx} '
                            f'but only {n_species} species are defined.'
                        )
        return data
