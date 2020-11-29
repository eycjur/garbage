from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

app_name="garbage"
urlpatterns = [
    path("", views.index, name="index"),
    path("result", views.result, name="result"),
    path("result/<int:num>", views.result, name="result_num"),
    path("search", views.search, name="search"),
    path("opinion", views.opinion, name="opinion"),
    path("opinion_submit", views.opinion_submit, name="opinion_submit"),
]