# -*- coding:utf-8 -*-
"""
author:Bird Christopher
date:2022//04//03
"""
from django.urls import path
from image_process.views import *

urlpatterns=[
    path("retrieval/",retrieval),
    path("contrast/",contrast),
    path("sort/",sort),
    path("detection/",detection)
]
