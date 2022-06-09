from django.shortcuts import render,redirect
from django.http import JsonResponse
from django.contrib.auth import authenticate,login
from django.contrib.auth.decorators import login_required,permission_required
from rest_framework import status
from rest_framework.decorators import *
from account.models import *
# from account.auth_backend import Email_auth
import inspect


# Create your views here.


@api_view(["POST"])
def login_view(request):
    email = request.data["email"]
    password = request.data["password"]
    user = authenticate(request,email = email,password = password)
    user.get_user_permissions()
    if user is not None:
        login(request, user)  # TODO: 这里可能需要重定向？？
        return JsonResponse({
            "code":status.HTTP_200_OK,
            "msg":"login successfully"
        })
    else:
        return JsonResponse({
            "code":status.HTTP_404_NOT_FOUND,
            "msg":"login failed"
        })


@api_view(["POST"])
def register_view(request):
    try:
        username = request.data["username"]
        password = request.data["password"]
        email = request.data["email"]
    except:
        return JsonResponse({
            "code":status.HTTP_406_NOT_ACCEPTABLE,
            "msg":"information not enough"
        })
    try:
        User.objects.create_user(username = username,password=password,email=email)
        return JsonResponse({
            "code":status.HTTP_200_OK
        })
    except:
        return JsonResponse({
            "code":status.HTTP_406_NOT_ACCEPTABLE,
            "msg":"exist duplicate username or email"
        })


