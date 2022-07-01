# -*- coding:utf-8 -*-
"""
author:Bird Christopher
date:2022//04//05
"""
from django.urls import path
from account.views import *
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

app_name = 'account'

urlpatterns = [
    path("register/",register_view,name="register_view"),
    path('token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
]
