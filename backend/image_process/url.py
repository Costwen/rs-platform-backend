# -*- coding:utf-8 -*-
"""
author:Bird Christopher
date:2022//04//03
"""
from django.urls import path
from image_process.views import *

urlpatterns=[
    path("new_task/",create_new_task,name="create_task"),
    path("task/change/<int:id>/", change_task_info, "change_tasks_info"),
    path("history/",get_specific_history,"get_all_task_history"),
    path("task/upload/<int:id>/", get_all_history,"get_one_task_history")
]
