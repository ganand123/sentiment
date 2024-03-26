# sentiment_analysis_project/sentiment_analysis_project/urls.py

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('sentiment_analysis.urls')),  # Include app's URLs
]


