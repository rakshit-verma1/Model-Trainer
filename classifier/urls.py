from django.urls import path
from . import views

urlpatterns = [
    path("",                           views.index,            name="index"),
    path("upload/",                    views.upload_dataset,   name="upload"),
    path("train/",                     views.start_train,      name="start_train"),
    path("stream/<str:session_id>/",   views.stream_progress,  name="stream"),
    path("download/<str:session_id>/", views.download_model,   name="download"),
    path("metrics/<str:session_id>/",  views.download_metrics, name="metrics"),
]
