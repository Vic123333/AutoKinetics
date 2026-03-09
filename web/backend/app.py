"""
AutoKinetics Web Backend
Flask API that exposes the chemical kinetics simulation engine.
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import traceback
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from data_model import Species, Reaction, ReactionSystem
from simulator import ODESolver
from analyzer import analyze_kinetics

app = Flask(
    __name__,
    static_folder=os.path.join(os.path.dirname(__file__), '..', 'frontend'),
    static_url_path=''
)
CORS(app)


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/simulate', methods=['POST'])
def simulate():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON payload received'}), 400

        # Build species list
        species_list = []
        for s in data.get('species', []):
            sp = Species(
                name=s['name'],
                start_concentration=float(s.get('start_concentration', 1.0)),
                is_intermediate=bool(s.get('is_intermediate', False))
            )
            species_list.append(sp)

        if not species_list:
            return jsonify({'success': False, 'error': 'At least one species is required'}), 400

        # Build reaction list
        reaction_list = []
        for r in data.get('reactions', []):
            reactants = [(int(item[0]), float(item[1])) for item in r['reactants']]
            products  = [(int(item[0]), float(item[1])) for item in r['products']]
            params = {
                'arrhenius_A':          float(r.get('arrhenius_A', 1.0)),
                'activation_energy_Ea': float(r.get('activation_energy_Ea', 0.0)),
                'temperature_exponent_n': float(r.get('temperature_exponent_n', 0.0)),
                'reaction_order': r.get('reaction_order', ''),
            }
            reaction = Reaction(reactants, products, r['rate_label'], **params)
            reaction_list.append(reaction)

        if not reaction_list:
            return jsonify({'success': False, 'error': 'At least one reaction is required'}), 400

        system      = ReactionSystem(species_list, reaction_list)
        temperature = float(data.get('temperature', 298.15))
        solver      = ODESolver(system, temperature)

        t_end    = float(data.get('t_end', 100.0))
        n_points = min(int(data.get('n_points', 400)), 2000)
        t_eval   = np.linspace(0, t_end, n_points)

        solution = solver.solve((0, t_end), t_eval)

        # Concentrations dict  {species_name: [float, ...]}
        concentrations = {}
        for i, sp in enumerate(species_list):
            concentrations[sp.name] = np.clip(solution.y[i], 0, None).tolist()

        # Analyzer uses flat indexed format
        sim_for_analyzer = {
            'time_points':    solution.t.tolist(),
            'concentrations': [np.clip(solution.y[i], 0, None).tolist()
                               for i in range(len(species_list))],
        }
        raw_analysis = analyze_kinetics(sim_for_analyzer, system)

        order_labels = {
            'zero_order':   '0th Order',
            'first_order':  '1st Order',
            'second_order': '2nd Order',
        }

        analysis = {}
        for rxn_label, info in raw_analysis.items():
            analysis[rxn_label] = {
                'analyzed_reactant': info['analyzed_reactant'],
                'best_fit_order':    order_labels.get(info['best_fit_order'], info['best_fit_order']),
                'calculated_k':      round(info['calculated_k'], 8),
                'k_unit':            info['k_unit'],
                'r_squared':         round(info['r_squared'], 6),
                'all_fits': {
                    order_labels.get(k, k): {
                        'r_squared': round(v['r_squared'], 6),
                        'k':         round(v['k'], 8),
                        'unit':      v['unit'],
                    }
                    for k, v in info['all_fits'].items()
                },
            }

        return jsonify({
            'success':        True,
            'time':           solution.t.tolist(),
            'concentrations': concentrations,
            'rate_equations': system.get_rate_law_equations(),
            'analysis':       analysis,
            'species_names':  [sp.name for sp in species_list],
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '1') == '1'
    app.run(debug=debug, port=port, host='0.0.0.0')
