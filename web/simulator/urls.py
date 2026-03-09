from django.urls import path
from . import views

urlpatterns = [
    path('',              views.index,        name='index'),
    path('api/simulate/', views.SimulateView.as_view(), name='simulate'),
    path('api/health/',   views.health,       name='health'),
]
