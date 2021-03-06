# -*- coding:utf-8 -*-
"""
author:Bird Christopher
date:2022//04//03
"""
from django.urls import path
from image_process.views import *

app_name = 'image_process'

urlpatterns=[
    path("project/", ProjectSetView.as_view(), name="create_project"),
    path("project/<pk>/", ProjectDetailView.as_view(), name="detail_project"),
    path("task/", TaskSetView.as_view(), name="create_task"),
    path("task/<pk>/", TaskDetailView.as_view(), name="detail_task"),
    path("image/", ImageUploadView.as_view(), name="image_upload"),
    path("image/<pk>/", ImageManagementView.as_view(), name="image_management")
]
