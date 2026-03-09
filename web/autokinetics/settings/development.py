from .base import *  # noqa

DEBUG = True
ALLOWED_HOSTS = ['*']

# Human-readable JSON in development
REST_FRAMEWORK['DEFAULT_RENDERER_CLASSES'] = [  # noqa
    'rest_framework.renderers.JSONRenderer',
    'rest_framework.renderers.BrowsableAPIRenderer',
]

# Relax throttle during development
REST_FRAMEWORK['DEFAULT_THROTTLE_RATES'] = {'anon': '300/minute'}  # noqa
