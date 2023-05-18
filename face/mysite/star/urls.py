from django.urls import path, include
from . import views
from star.views import classify_image
from .views import rank

urlpatterns = [
    # path('', views.index) ,
    #path('', views.index, name='index'),
    path('classify/', classify_image, name='classify_image'),
    path('ranking/', rank, name='rank'),
]
