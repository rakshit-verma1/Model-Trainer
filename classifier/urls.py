from django.urls import path
from . import views

urlpatterns = [
    path("",                           views.index,            name="index"),
    path("upload/",                    views.upload_dataset,   name="upload"),
    path("train/",                     views.start_train,      name="start_train"),
    path("stream/<str:session_id>/",   views.stream_progress,  name="stream"),
    path("download/<str:session_id>/", views.download_model,   name="download"),
    path("metrics/<str:session_id>/",  views.download_metrics, name="metrics"),
    # Gemini AI endpoints
    path("api/gemini/suggest/",        views.gemini_suggest,      name="gemini_suggest"),
    path("api/gemini/upload-pdf/",     views.gemini_upload_pdf,   name="gemini_upload_pdf"),
    path("api/gemini/help-layer/",     views.gemini_help_layer,   name="gemini_help_layer"),
]
