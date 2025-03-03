from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # Homepage URL
    path('analyze-url/', views.analyze_url, name='analyze_url'),
    path('analyze-file/', views.analyze_file, name='analyze_file'),
    path('check-task-status/', views.check_task_status, name='check_task_status'),
]