# -*- coding:utf-8 -*-
"""
author:Bird Christopher
date:2022//04//03
"""
from django.urls import path
from image_process.views import *

urlpatterns=[
    path("project/", ProjectSetView.as_view(), name="create_project"),
    path("project/<pk>/", ProjectDetailView.as_view(), name="detail_project"),
    path("image/", ImageUploadView.as_view(), name="image_upload"),
    path("image/<int:pk>/", ImageManagementView.as_view(), name="image_management")
]
