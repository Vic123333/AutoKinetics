# AutoKinetics Web

A scientifically rigorous, production-ready **chemical reaction kinetics simulator** that runs in the browser. Built with Django + Django REST Framework; no desktop software required.

---

## Scientific Foundations

| Feature | Implementation |
|---|---|
| Rate constant | Modified Arrhenius: `k(T) = A·Tⁿ·exp(−Eₐ/RT)` (IUPAC) |
| ODE system | `d[Xᵢ]/dt = Σⱼ νᵢⱼ·vⱼ` (stoichiometric matrix form) |
| Solver | **Radau** (implicit R-K, A-stable, 5th order); `rtol=10⁻⁷, atol=10⁻¹⁰` |
| QSSA | Tikhonov–Fenichel algebraic constraint `d[Xᵢ]/dt = 0` solved via Newton |
| Kinetic order | Linear regression on `[A]`, `ln[A]`, `1/[A]` vs `t` with R²/RMSE |
| Unit labels | Derived from stoichiometric order (IUPAC): s⁻¹, L mol⁻¹ s⁻¹, … |
| Half-life | `t½ = ln2/k` for 1st-order reactions |
| R² threshold | 0.98 (textbook standard; LibreTexts / Atkins PChem) |

Reference: Atkins, *Physical Chemistry*, 11th ed., Ch. 17; NIST Chemical Kinetics Database.

---

## Quick Start (Development)

```bash
cd web
python -m venv .venv && source .venv/bin/activate
pip install -r requirements/development.txt

cp .env.example .env   # edit SECRET_KEY at minimum

python manage.py runserver
# → http://127.0.0.1:8000
```

---

## Production Deployment

### Option A – Docker Compose (recommended)

```bash
cp .env.example .env
# Set SECRET_KEY, ALLOWED_HOSTS, DEBUG=False

docker compose up -d
```

Put your TLS certificates in `nginx/certs/fullchain.pem` and `nginx/certs/privkey.pem`
(or use Certbot: `certbot certonly --nginx -d example.com`).

### Option B – Bare-metal with Gunicorn + systemd

```bash
pip install -r requirements/production.txt
python manage.py collectstatic --settings=autokinetics.settings.production

# Generate a .env and export variables, then:
DJANGO_SETTINGS_MODULE=autokinetics.settings.production \
  gunicorn --config gunicorn.conf.py autokinetics.wsgi:application
```

---

## Security Features

| Layer | Mechanism |
|---|---|
| Input validation | DRF serializers with strict type/range checks |
| Rate limiting | `AnonRateThrottle` (30 req/min, configurable) |
| Computation timeout | `ThreadPoolExecutor` with 30 s hard limit |
| CSRF | Django CSRF middleware; token sent in API requests |
| Security headers | HSTS, CSP, X-Frame-Options, X-Content-Type-Options |
| TLS | nginx terminates HTTPS (TLSv1.2/1.3 only) |
| Nginx rate limit | 30 req/s per IP at proxy level, burst=10 |
| Container | Read-only filesystem, all capabilities dropped, `no-new-privileges` |
| Secrets | `SECRET_KEY` from env; production settings reject insecure default |

---

## API

### `POST /api/simulate/`

**Request** (JSON):
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
      "arrhenius_A": 1e10,
      "activation_energy_Ea": 60000,
      "temperature_exponent_n": 0.0
    }
  ],
  "temperature": 450,
  "t_end": 300,
  "n_points": 400
}
```

**Response** includes: `time`, `concentrations`, `rate_equations`, `analysis` (R², k, best order, t½).

### `GET /api/health/`
Returns `{"status": "ok", "version": "2.0"}`.

---

## Project Structure

```
web/
├── manage.py
├── autokinetics/            Django project
│   ├── settings/
│   │   ├── base.py          Shared settings (env-driven)
│   │   ├── development.py
│   │   └── production.py    Adds HSTS, HTTPS, secret-key guard
│   ├── urls.py
│   └── wsgi.py
├── simulator/               Django app
│   ├── views.py             SimulateView (throttled, timeout-protected)
│   ├── serializers.py       Strict DRF validation
│   ├── urls.py
│   ├── kinetics/
│   │   ├── data_model.py    Species, Reaction, ReactionSystem
│   │   ├── solver.py        ODESolver (Radau, QSSA)
│   │   └── analyzer.py      Kinetic order analysis
│   └── templates/index.html Single-page app (Tailwind + Plotly.js)
├── requirements/
│   ├── base.txt
│   ├── development.txt
│   └── production.txt
├── Dockerfile
├── docker-compose.yml
├── gunicorn.conf.py
├── nginx/nginx.conf
└── .env.example
```
