# -*- coding:utf-8 -*-
"""
author:Bird Christopher
date:2022//04//03
"""
from django.urls import path
from image_process.views import *

urlpatterns=[
    path("retrieval/",retrieval,name="target_retrieval"),
    path("contrast/",contrast,name="change_contrast"),
    path("sort/",sort,name="plain_object_sort"),
    path("detection/",detection,name="object_detection")
]
