"""assign1 URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
# urls.py
from django.urls import path,include
from .views import views1, views2, views3, views5 , views6, views7, views8

urlpatterns = [
    # ... other URL patterns
    path('api/upload-csv/', views1.CSVUploadView.as_view()),
    path('api/chi2_analyze/', views2.Chi_Analyze.as_view()),
    path('api/corelation/', views2.Corelation_Analyze.as_view()),
    path('api/perform_normalization/', views2.PerformNormalization.as_view()),
    path('api/info_gain/', views3.InfoGain.as_view()),
    path('api/regression/', views5.Regression.as_view()), 
    path('api/clustering/', views6.Clustering.as_view()),
    
    path('api/crawler/', views8.Crawler.as_view()),

    path('api/run_association_rules/', views7.run_association_rules, name='run_association_rules'),
    path('api/run_association_rules_matrics/', views7.run_association_rules_matrics, name='run_association_rules'),
    
]
