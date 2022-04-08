# -*- coding:utf-8 -*-
"""
author:Bird Christopher
date:2022//04//04
"""
from django.contrib.auth.hashers import check_password
from django.contrib.auth.backends import BaseBackend
from django.contrib.auth import get_user_model
from django.core.exceptions import PermissionDenied
from django.db.models.query import QuerySet


class Email_auth(BaseBackend):
    def authenticate(self, request,email=None,password=None,**kwargs):
        try:
            User = get_user_model()
            user = User.objects.get(email=email)  # 如何使用覆盖索引,只能使用原生SQL？效率评估
            if user.check_password(password):
                return user
        except:
            print("user does not exist")
            return
        return

    def get_user(self,user_id):
        try:
            return get_user_model().objects.get(pk = user_id)
        except:
            return None

    def has_perm(self,user_obj,perm,obj = None):
        print("has perm yoh!")
        return True

    def get_user_permissions(self, user_obj, obj=None):
        # 定义着玩的
        return "has all permissions"
