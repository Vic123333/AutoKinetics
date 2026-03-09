"""
Django REST Framework views for AutoKinetics.

Security features:
  • DRF AnonRateThrottle  (configurable, default 30 req/min per IP)
  • Strict DRF serializer validation on all inputs
  • Hard computation timeout via ThreadPoolExecutor
  • No shell execution, no file-system access from user input
  • CSRF required for browser-originated POST requests
"""
from __future__ import annotations
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

import numpy as np
from django.conf import settings
from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view, throttle_classes
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.throttling import AnonRateThrottle
from rest_framework.views import APIView, exception_handler

from .serializers import SimulationRequestSerializer
from .kinetics.data_model import Species, Reaction, ReactionSystem
from .kinetics.solver import ODESolver
from .kinetics.analyzer import analyze_kinetics

log = logging.getLogger(__name__)


# ── Custom exception handler ─────────────────────────────────────────────────
def custom_exception_handler(exc, context):
    response = exception_handler(exc, context)
    if response is not None:
        return response
    log.exception('Unhandled exception in view')
    return Response(
        {'detail': 'An internal error occurred. Please try again.'},
        status=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


# ── Frontend view ─────────────────────────────────────────────────────────────
def index(request):
    return render(request, 'index.html')


# ── Simulation API ────────────────────────────────────────────────────────────
class SimulateView(APIView):
    throttle_classes = [AnonRateThrottle]

    def post(self, request: Request):
        # 1. Validate input
        ser = SimulationRequestSerializer(data=request.data)
        if not ser.is_valid():
            return Response({'detail': 'Validation error', 'errors': ser.errors},
                            status=status.HTTP_400_BAD_REQUEST)

        data = ser.validated_data

        # 2. Run simulation in sandboxed thread with hard timeout
        timeout = getattr(settings, 'SIMULATION_TIMEOUT_S', 30)
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_run_simulation, data)
                result = future.result(timeout=timeout)
        except FutureTimeout:
            log.warning('Simulation timed out after %ds', timeout)
            return Response(
                {'detail': f'Simulation exceeded the {timeout}s time limit. '
                           'Try reducing t_end or the number of species.'},
                status=status.HTTP_408_REQUEST_TIMEOUT,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception('Simulation error')
            return Response(
                {'detail': f'Simulation failed: {exc}'},
                status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        if not result.get('success'):
            return Response(
                {'detail': result.get('error', 'Solver did not converge.')},
                status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        return Response(result, status=status.HTTP_200_OK)


# ── Core computation (runs in worker thread) ──────────────────────────────────
def _run_simulation(data: dict) -> dict:
    """Build the kinetic model, integrate, analyse.  No I/O, no side-effects."""
    species_list = [
        Species(
            name=s['name'],
            start_concentration=float(s['start_concentration']),
            is_intermediate=bool(s['is_intermediate']),
        )
        for s in data['species']
    ]

    reaction_list = [
        Reaction(
            reactants=[(int(idx), float(s)) for idx, s in r['reactants']],
            products= [(int(idx), float(s)) for idx, s in r['products']],
            rate_label=r['rate_label'],
            arrhenius_A=r['arrhenius_A'],
            activation_energy_Ea=r['activation_energy_Ea'],
            temperature_exponent_n=r['temperature_exponent_n'],
            reaction_order=r.get('reaction_order', {}),
        )
        for r in data['reactions']
    ]

    system = ReactionSystem(species_list, reaction_list)
    solver = ODESolver(system, temperature=float(data['temperature']))

    t_end    = float(data['t_end'])
    n_points = int(data['n_points'])
    t_eval   = np.linspace(0.0, t_end, n_points)

    sol = solver.solve((0.0, t_end), t_eval)

    if not sol.success:
        return {'success': False,
                'error': f'ODE solver failed: {sol.message}. '
                          'Consider tightening initial concentrations or reducing t_end.'}

    # Clip negatives (numerical noise near zero)
    conc_dict = {
        sp.name: np.clip(sol.y[i], 0.0, None).tolist()
        for i, sp in enumerate(species_list)
    }

    # Kinetic order analysis
    sim_for_analyzer = {
        'time_points':    sol.t.tolist(),
        'concentrations': [np.clip(sol.y[i], 0.0, None).tolist()
                           for i in range(len(species_list))],
    }
    analysis = analyze_kinetics(sim_for_analyzer, system)

    # Human-friendly format for JSON response
    analysis_out = {}
    order_label  = {'zero_order': '0th Order', 'first_order': '1st Order',
                    'second_order': '2nd Order'}
    for rxn_lbl, info in analysis.items():
        fits = {
            order_label.get(k, k): {
                'r_squared': round(v['r_squared'], 6),
                'k':         round(v['k'], 8),
                'unit':      v['unit'],
                'rmse':      round(v['rmse'], 8),
            }
            for k, v in info['all_fits'].items()
        }
        entry = {
            'analyzed_reactant': info['analyzed_reactant'],
            'best_fit_order':    order_label.get(info['best_fit_order'], info['best_fit_order']),
            'calculated_k':      round(info['calculated_k'], 8),
            'k_unit':            info['k_unit'],
            'r_squared':         round(info['r_squared'], 6),
            'fits_well':         info['fits_well'],
            'all_fits':          fits,
        }
        if 'half_life_s' in info:
            entry['half_life_s']           = info['half_life_s']
            entry['characteristic_time_s'] = info['characteristic_time_s']
        analysis_out[rxn_lbl] = entry

    return {
        'success':        True,
        'time':           sol.t.tolist(),
        'concentrations': conc_dict,
        'species_names':  [sp.name for sp in species_list],
        'rate_equations': system.get_rate_law_equations(),
        'analysis':       analysis_out,
        'solver_message': sol.message,
    }


# ── Health check ──────────────────────────────────────────────────────────────
@api_view(['GET'])
@throttle_classes([])
def health(request):
    return Response({'status': 'ok', 'version': '2.0'})
