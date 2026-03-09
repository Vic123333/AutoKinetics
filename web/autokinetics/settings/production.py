from .base import *  # noqa

# ---------- Security hardening for production ----------
SECURE_SSL_REDIRECT         = True
SESSION_COOKIE_SECURE       = True
CSRF_COOKIE_SECURE          = True
SECURE_HSTS_SECONDS         = 31_536_000   # 1 year
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD         = True
SECURE_PROXY_SSL_HEADER     = ('HTTP_X_FORWARDED_PROTO', 'https')

# Require SECRET_KEY to be set in production environment
import os
if os.environ.get('SECRET_KEY', '').startswith('django-insecure'):
    raise RuntimeError(
        'FATAL: SECRET_KEY must be set to a secure value in production. '
        'Generate one with: python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"'
    )
