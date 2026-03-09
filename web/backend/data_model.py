import numpy as np

R = 8.31446261815324  # Universal gas constant in J/(mol·K)

class Species:
    """Represents a single chemical species with its properties."""
    def __init__(self, name, **kwargs):
        self.name = name
        self.start_concentration = float(kwargs.get('start_concentration', 1.0))
        self.concentration = self.start_concentration
        
        for key, value in kwargs.items():
            setattr(self, key, value)

class Reaction:
    """Represents a single reaction with its kinetic parameters."""
    def __init__(self, reactants, products, rate_label, **params):
        self.reactants = reactants
        self.products = products
        self.rate_label = rate_label
        
        self.arrhenius_A = float(params.get('arrhenius_A', 1.0))
        self.activation_energy_Ea = float(params.get('activation_energy_Ea', 0.0))
        self.temp_exponent_n = float(params.get('temperature_exponent_n', 0.0))
        
        self.reaction_order = {}
        try:
            # Try to read an explicit overall reaction order from the file.
            order_val = float(params.get('reaction_order', ''))
            for r_idx, _ in self.reactants:
                self.reaction_order[r_idx] = order_val
        except (ValueError, TypeError):
            # If 'reaction_order' is not a valid number (e.g., empty string),
            # the default behavior is to use the stoichiometry of each reactant as its partial order.
            reactant_counts = {}
            for r_idx, stoich_in_tuple in self.reactants:
                reactant_counts[r_idx] = reactant_counts.get(r_idx, 0) + stoich_in_tuple
            
            for r_idx, total_stoich in reactant_counts.items():
                self.reaction_order[r_idx] = total_stoich

    def calculate_k(self, T):
        """Calculates the rate constant k at temperature T."""
        if self.arrhenius_A == 0: return 0.0
        Ea_J_mol = self.activation_energy_Ea
        k = self.arrhenius_A * (T ** self.temp_exponent_n) * np.exp(-Ea_J_mol / (R * T))
        return k

class ReactionSystem:
    """Manages the entire system of species and reactions."""
    def __init__(self, species_list, reaction_list):
        self.species = species_list
        self.reactions = reaction_list
        self.species_map = {s.name: i for i, s in enumerate(species_list)}

    def get_initial_concentrations(self):
        return np.array([s.start_concentration for s in self.species])

    def get_rate_law_equations(self):
        """Creates a correct textual representation of the differential rate law."""
        equations = []
        species_names = [s.name for s in self.species]
        
        for i, species in enumerate(self.species):
            lhs = f"d[{species_names[i]}]/dt ="
            rhs_terms = []

            for reaction in self.reactions:
                rate_expression_parts = [reaction.rate_label]
                
                reactant_orders = {}
                for reactant_idx, _ in reaction.reactants:
                    # ** BUG FIX IS HERE **
                    # Was: self.reaction_order.get(...)
                    # Now: reaction.reaction_order.get(...)
                    order = reaction.reaction_order.get(reactant_idx, 1.0)
                    reactant_orders[reactant_idx] = order
                
                for reactant_idx, order in sorted(reactant_orders.items()):
                    order_str = f"^{order}" if order != 1.0 else ""
                    rate_expression_parts.append(f"[{species_names[reactant_idx]}]" + order_str)

                rate_expr = " * ".join(rate_expression_parts)

                net_stoichiometry = 0
                for reactant_idx, stoich in reaction.reactants:
                    if reactant_idx == i:
                        net_stoichiometry -= stoich
                for product_idx, stoich in reaction.products:
                    if product_idx == i:
                        net_stoichiometry += stoich

                if net_stoichiometry != 0:
                    if net_stoichiometry > 0:
                        sign = "+"
                    else:
                        sign = "-"
                    
                    abs_stoich = abs(net_stoichiometry)
                    stoich_str = f"{abs_stoich} * " if abs_stoich != 1 else ""
                    rhs_terms.append(f"{sign} {stoich_str}{rate_expr}")
            
            if not rhs_terms:
                rhs = " 0.0"
            else:
                rhs = " ".join(sorted(rhs_terms))
                if rhs.startswith("+ "):
                    rhs = rhs[2:]
                elif rhs.startswith("- "):
                    rhs = rhs[0] + rhs[2:]

            equations.append(lhs + rhs)
            
        return "\n".join(equations)