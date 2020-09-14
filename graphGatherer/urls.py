from django.urls import path

from . import views

app_name = 'graphGatherer'

urlpatterns = [
    #ex /graphs/index
    path('index', views.main_index, name='index'),
    #ex /graphs/thresholds
    path('<str:dev_id>/thresholds', views.thresholds, name='thresholds'),
    #ex /graphs/errors
    path('<str:dev_id>/errors', views.errors, name='errors'),
    #ex /graphs/DAC123/Tamb
    path('<str:dev_id>/', views.display_graph, name='display_graph'),
]