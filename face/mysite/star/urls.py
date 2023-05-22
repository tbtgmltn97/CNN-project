from django.urls import path, include
from . import views
from star.views import classify_image
from .views import rank

app_name = 'star'

urlpatterns = [
    path('', classify_image, name='classify_image'),
    path('ranking/', rank, name='rank'),
    path('sub_menu_1/', views.sub_menu_1, name='sub_menu_1'), #3.35.235.162:8000/sub_menu_1로 가면 sub_menu_1함수로!
]
