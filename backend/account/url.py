# -*- coding:utf-8 -*-
"""
author:Bird Christopher
date:2022//04//05
"""
from django.urls import path
from account.views import *

urlpatterns = [
    path("login/",login_view,name="login_view"),
    path("register/",register_view,name="register_view"),
    path("workspace/",view_workspace,name="workspace_view"),
    path("redirect/",try_to_redirect,name="redirect_view")
]
