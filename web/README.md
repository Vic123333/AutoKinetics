# AutoKinetics Web

A browser-based chemical reaction kinetics simulator. No installation of PyQt6 or desktop software needed — runs entirely in the browser via a lightweight Flask API.

## Features

- **8 built-in presets**: 1st/2nd order, consecutive, competitive, Arrhenius, Michaelis-Menten enzyme kinetics, ozone photodecomposition, autocatalytic
- **Visual reaction network diagram** (SVG, auto-layout)
- **Interactive Plotly charts** — zoom, pan, hover, download as PNG
- **Rate law equations** — displays the ODE system being solved
- **Kinetic order analysis** — fits 0th/1st/2nd order and shows R² for each
- **QSSA support** — Quasi-Steady-State Approximation for reactive intermediates
- **CSV export** of simulation results
- **Arrhenius kinetics** — full k(T) = A·Tⁿ·exp(−Eₐ/RT) support

## Quick Start

```bash
cd web
pip install -r requirements.txt
python backend/app.py
# Open http://localhost:5000
```

## Production

```bash
gunicorn -w 4 -b 0.0.0.0:8000 backend.app:app
```

## API

### `POST /api/simulate`

```json
{
  "species": [
    {"name": "A", "start_concentration": 1.0, "is_intermediate": false},
    {"name": "B", "start_concentration": 0.0}
  ],
  "reactions": [
    {
      "rate_label": "k₁",
      "reactants": [[0, 1]],
      "products":  [[1, 1]],
      "arrhenius_A": 0.1,
      "activation_energy_Ea": 0,
      "temperature_exponent_n": 0
    }
  ],
  "temperature": 298.15,
  "t_end": 60
}
```

Response includes `time`, `concentrations`, `rate_equations`, and `analysis`.
