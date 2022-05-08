from statistics import mode
import time

from matplotlib import image
from requests import delete
from rest_framework import status
from rest_framework.decorators import api_view
from backend.settings import predictor as P
from backend.util import MapImageHelper
from django.contrib.auth.models import AnonymousUser
from django.contrib.auth import get_user_model
import PIL
from image_process.models import *
from django.shortcuts import get_object_or_404
from image_process.tasks import *
from celery.result import AsyncResult
from image_process.tasks import image_handler
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework.response import Response
from rest_framework.views import APIView


# TODO： account not done
@api_view(["PUT"])
def create_new_task(request):
    project_id = request.data.get("project_id", None)
    user_id = request.user.pk
    coordinate = request.data.get("coordinate",None)
    task = image_handler.delay(project_id, user_id, coordinate)
    Task.objects.create(task_id=task.id, user_id=user_id, project_id=project_id, coordinate=coordinate)
    return Response(
        data={
            "message": "创建成功",
            "task_id": task.id},
        status=status.HTTP_200_OK
    )


@api_view(["POST","DEL"])
def change_task_info(request):
    pass


@api_view(["GET"])
def result_detail(request):
    user = request.user
    result = get_object_or_404(Inference,user = user,pk = request.GET["id"])
    return JsonResponse

def login_required(func):
    def wrapper(self, request,*args,**kwargs):
        if isinstance(request.user,AnonymousUser):
            return Response(
                data={"message":"请登录后再操作"},
                status=status.HTTP_401_UNAUTHORIZED
            )
        return func(self, request,*args,**kwargs)
    return wrapper


class ProjectSetView(APIView):
    # 限制请求方式
    http_method_names = ["get", "put"]

    @login_required
    def get(self,request):
        user = request.user
        projects = Project.objects.filter(user = user)
        results = []
        for project in projects:
            results.append({
                "name": project.name,
                "type": project.type,
                "imageA": project.imageA,
                "imageB": project.imageB,
                "id": project.pk,
                "create_time": project.create_time,
            })
        return Response(
            data={"message":"获取成功","projects":results},
            status=status.HTTP_200_OK
        )
    @login_required
    def put(self,request):
        user = request.user
        imageA = request.data.get("imageA","")
        imageB = request.data.get("imageB","")
        name = request.data.get("name","")
        type = request.data.get("type","")
        project = Project.objects.create(user = user,name = name,type = type, imageA = imageA, imageB = imageB)
        return Response(
            data={"message":"创建成功","project_id":project.pk},
            status=status.HTTP_200_OK
        )

class ProjectDetailView(APIView):
    # 限制请求方式
    http_method_names = ["get", "post", "delete"]
    @login_required
    def get(self,request, pk):
        user = request.user
        project = get_object_or_404(Project,user = user,pk = pk)
        data = {
            "name": project.name,
            "type": project.type,
            "imageA": project.imageA,
            "imageB": project.imageB,
            "id": project.pk,
            "create_time": project.create_time,
        }
        return Response(
            data={"message":"获取成功","project":data},
            status=status.HTTP_200_OK
        )

    @login_required
    def delete(self,request, pk):
        user = request.user
        project = get_object_or_404(Project,user = user,pk = pk)
        project.delete()
        return Response(
            data={"message":"删除成功"},
            status=status.HTTP_200_OK
        )
    @login_required
    def post(self,request, pk):
        user = request.user
        project = get_object_or_404(Project,user = user,pk = pk)
        project.update(**request.data)
        return Response(
            data={"message":"更新成功"},
            status=status.HTTP_200_OK
        )
