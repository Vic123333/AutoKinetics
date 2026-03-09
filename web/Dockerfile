FROM python:3.12-slim

# Non-root user for security
RUN useradd --create-home --shell /bin/bash app

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements/production.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .
RUN python manage.py collectstatic --noinput \
        --settings=autokinetics.settings.production

# Permissions
RUN chown -R app:app /app
USER app

EXPOSE 8000

# Gunicorn – 2 workers per CPU (adjust via GUNICORN_WORKERS env var)
CMD ["gunicorn", \
     "--config", "gunicorn.conf.py", \
     "autokinetics.wsgi:application"]
