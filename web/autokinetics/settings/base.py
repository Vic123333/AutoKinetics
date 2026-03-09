"""
AutoKinetics – Django base settings.
All environment-specific values are pulled from the .env file via django-environ.
"""
import environ
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

env = environ.Env(
    DEBUG=(bool, False),
    ALLOWED_HOSTS=(list, ['localhost', '127.0.0.1']),
    SIMULATION_TIMEOUT_S=(int, 30),
    MAX_SPECIES=(int, 20),
    MAX_REACTIONS=(int, 50),
    THROTTLE_RATE=(str, '30/minute'),
)
environ.Env.read_env(BASE_DIR / '.env', overwrite=False)

SECRET_KEY = env('SECRET_KEY', default='django-insecure-CHANGE-ME-in-production')
DEBUG = env('DEBUG')
ALLOWED_HOSTS = env('ALLOWED_HOSTS')

# ---------- Apps ----------
INSTALLED_APPS = [
    'django.contrib.staticfiles',
    'rest_framework',
    'simulator',
]

# ---------- Middleware ----------
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'autokinetics.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
            ],
        },
    },
]

WSGI_APPLICATION = 'autokinetics.wsgi.application'

# No database needed – this is a stateless computation service.
DATABASES = {}

# ---------- Static files ----------
STATIC_URL  = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ---------- Internationalisation ----------
LANGUAGE_CODE = 'en-us'
TIME_ZONE     = 'UTC'
USE_I18N      = True
USE_TZ        = True

# ---------- Django REST Framework ----------
REST_FRAMEWORK = {
    'DEFAULT_RENDERER_CLASSES': ['rest_framework.renderers.JSONRenderer'],
    'DEFAULT_THROTTLE_CLASSES':  ['rest_framework.throttling.AnonRateThrottle'],
    'DEFAULT_THROTTLE_RATES':    {'anon': env('THROTTLE_RATE')},
    'EXCEPTION_HANDLER': 'simulator.views.custom_exception_handler',
}

# ---------- Security ----------
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS             = 'DENY'
SECURE_REFERRER_POLICY      = 'strict-origin-when-cross-origin'

# ---------- Application-level limits ----------
SIMULATION_TIMEOUT_S = env('SIMULATION_TIMEOUT_S')
MAX_SPECIES          = env('MAX_SPECIES')
MAX_REACTIONS        = env('MAX_REACTIONS')
