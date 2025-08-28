from django.urls import path
from .views import RecipePredictView

urlpatterns = [
    path('predict/', RecipePredictView.as_view(), name='predict'),
]
