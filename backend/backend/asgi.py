"""
ASGI config for backend project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.1/howto/deployment/asgi/
"""

import os
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.dev")
django.setup()
from channels.routing import ProtocolTypeRouter, URLRouter
from image_process.routing import websocket_urlpatterns
from django.core.asgi import get_asgi_application
from channels.security.websocket import AllowedHostsOriginValidator
from .channelsmiddleware import JwtAuthMiddlewareStack



application = ProtocolTypeRouter(
    {
        "http": get_asgi_application(),
        "websocket": AllowedHostsOriginValidator(
            JwtAuthMiddlewareStack(
                URLRouter(websocket_urlpatterns)
            ),
        ),
    }
)