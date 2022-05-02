import django.contrib.sessions.backends.db
from django.shortcuts import render,redirect
from django.http import JsonResponse
from django.contrib.auth import authenticate,login
from django.contrib.auth.backends import ModelBackend
from django.contrib.auth.decorators import login_required,permission_required
from rest_framework import status
from rest_framework.decorators import *
from account.models import *
# from account.auth_backend import Email_auth
import inspect
# Create your views here.


# TODO: 用Redis存储所有的session
@api_view(["POST"])
def login_view(request):
    email = request.data["email"]
    password = request.data["password"]
    user = authenticate(request,email = email,password = password)
    user.get_user_permissions()
    if user is not None:
        login(request, user)
        # TODO: do something, may be redirecting to original view?
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


# TODO:缩略图的传输..
@login_required
def view_workspace(request):
    user = request.user
    page_length = request.data["length"]
    inferences = user.inference_set.all()
    results = []
    for inference in inferences:
        results.append({
            "raw_image_url":request.scheme+"://"+request.META["HTTP_HOST"]+"/images/"+inference.raw.name,
            "upload_time":inference.upload_time,
            "task":inference.task,
            "name":inference.name,
            "id":inference.pk
        })
    return JsonResponse({
        "code":status.HTTP_200_OK,
        "results":results
    })

@api_view(["POST"])
def try_to_redirect(request):
    return redirect("workspace_view")


