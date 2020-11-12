from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

app_name="garbage"
urlpatterns = [
    path("", views.index, name="index"),
    path("result", views.result, name="result"),
    path("sample1", views.sample1, name="sample1"),
    path("sample2", views.sample2, name="sample2"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)