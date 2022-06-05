import ast
from hashlib import new
import json
from matplotlib import image
from rest_framework import status
from rest_framework.decorators import api_view
from sympy import re
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
import demjson


# # TODO： account not done
# @api_view(["PUT"])
# def create_new_task(request):
#     project_id = request.data.get("project_id", None)
#     user_id = request.user.pk
#     coordinate = request.data.get("coordinate",None)
#     task = image_handler.delay(project_id, user_id, coordinate)
#     Task.objects.create(task_id=task.id, user=user, project=project, coordinate=coordinate)
#     return Response(
#         data={
#             "message": "创建成功",
#             "task_id": task.id},
#         status=status.HTTP_200_OK
#     )


# @api_view(["POST","DEL"])
# def change_task_info(request):
#     pass


# @api_view(["GET"])
# def result_detail(request):
#     user = request.user
#     result = get_object_or_404(Inference,user = user,pk = request.GET["id"])
#     return JsonResponse

# 验证登录
# 跳转到登录界面更好
def login_required(func):
    def wrapper(self, request,*args,**kwargs):
        if isinstance(request.user,AnonymousUser):
            return Response(
                data={"message":"请登录后再操作"},
                status=status.HTTP_401_UNAUTHORIZED
            )
        return func(self, request,*args,**kwargs)
    return wrapper


# 任务类
class TaskSetView(APIView):
    http_method_names = ["get", "post"]

    @login_required
    def post(self,request):
        user = request.user
        project_id = request.data.get("project_id", None)
        coordinate = request.data.get("coordinate",None)
        task = Task.objects.create(user=user, project_id=project_id, coordinate=coordinate)
        image_handler.delay(task_id=task.id)
        return Response(
            data={"message":"创建成功","id":task.pk},
            status=status.HTTP_200_OK
        )
    
    @login_required
    def get(self,request):
        user = request.user
        tasks = Task.objects.filter(user=user)
        data = []
        for task in tasks:
            data.append({
                "id":task.pk,
                "project_id":task.project_id,
                "coordinate":task.coordinate,
                "status":task.status,
                "mask":task.mask,
                "create_time":task.create_time,
            })
        return Response(
            data={"message":"获取成功","data":data},
            status=status.HTTP_200_OK
        )

# 任务详情类
class TaskDetailView(APIView):
    http_method_names = ["get", "delete"]

    @login_required
    def get(self,request, pk):
        user = request.user
        task = get_object_or_404(Task,user = user,pk = pk)
        return Response(
            data={"message":"获取成功","data":{
                "id":task.pk,
                "project_id":task.project_id,
                "coordinate":task.coordinate,
                "status":task.status,
                "mask":task.mask,
                "create_time":task.create_time,
            }},
            status=status.HTTP_200_OK
        )

    @login_required
    def delete(self,request, pk):
        user = request.user
        task = get_object_or_404(Task,user = user,pk = pk)
        task.delete()
        return Response(
            data={"message":"删除成功"},
            status=status.HTTP_200_OK
        )



# 项目类
class ProjectSetView(APIView):
    # 限制请求方式
    http_method_names = ["get", "post"]

    @login_required
    def get(self,request):
        param = request.query_params
        _status = param.get("status",None)
        user = request.user
        projects = Project.objects.filter(user = user)
        if _status:
            projects = projects.filter(status = _status)
        results = []
        for project in projects:
            tasks = Task.objects.filter(user = user, project_id = project.pk)
            task_num = tasks.count()
            # 格式化时间
            create_time = project.create_time.strftime("%Y-%m-%d %H:%M:%S")
            modify_time = project.modify_time.strftime("%Y-%m-%d %H:%M:%S")
            results.append({
                "name": project.name,
                "type": project.type,
                "imageA": project.imageA,
                "imageB": project.imageB,
                "id": project.pk,
                "create_time": create_time,
                "modify_time": modify_time,
                "task_num": task_num,
                "status": project.status,
            })
        return Response(
            data={"message":"获取成功","projects":results},
            status=status.HTTP_200_OK
        )

    @login_required
    def post(self,request):
        user = request.user
        imageA = request.data.get("imageA","")
        imageB = request.data.get("imageB","")
        name = request.data.get("name","")
        type = request.data.get("type","")
        project = Project.objects.create(user = user,name = name,type = type, imageA = imageA, imageB = imageB)
        return Response(
            data={"message":"创建成功","id":project.pk},
            status=status.HTTP_200_OK
        )


# 项目详情类
class ProjectDetailView(APIView):
    # 限制请求方式
    http_method_names = ["get", "put", "delete"]
    @login_required
    def get(self,request, pk):
        user = request.user
        project = get_object_or_404(Project,user = user,pk = pk)
        tasklist = Task.objects.filter(project_id = project.pk)
        tasks = []
        for task in tasklist:
            tasks.append({
                "id":task.pk,
                "project_id":task.project_id,
                "coordinate":task.coordinate,
                "status":task.status,
                "mask":task.mask,
                "create_time":task.create_time,
            })
        data = {
            "name": project.name,
            "type": project.type,
            "imageA": project.imageA,
            "imageB": project.imageB,
            "id": project.pk,
            "create_time": project.create_time,
            "tasks": tasks,     
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
    def put(self,request, pk):
        user = request.user
        project = Project.objects.filter(pk=pk)
        if len(project) == 0:
            return Response(
                data={"message":"项目不存在"},
                status=status.HTTP_400_BAD_REQUEST
            )
        project.update(**request.data)
        return Response(
            data={"message":"更新成功"},
            status=status.HTTP_200_OK
        )


class ImageUploadView(APIView):

    http_method_names = ["post","get","put"]
    # 用户上传图片 from 高德
    @login_required
    def post(self, request):
        coordinate = request.data["coordinate"]
        name = request.data.get("name","Untitled")
        #  convert coordinate to json
        new_record = Image(user = request.user, name=name, type = "map")
        coordinate = json.loads(coordinate)
        target_img = MapImageHelper.getImage(coordinate["tl"][0],coordinate["br"][1],coordinate["br"][0],coordinate["tl"][1])
        filename = new_record.id + ".png"
        target_img.save("./media/"+filename)
        new_record.url = "/images/" + filename
        new_record.save()
        return Response(
            status=status.HTTP_200_OK,
            data={
                "url":request.scheme+"://"+request.META["HTTP_HOST"]+new_record.url,
                "id" :new_record.pk,
                "create_time":new_record.create_time,
                "name": new_record.name
            }
        )

    # 用户获得自己的所有图片
    @login_required
    def get(self, request):
        images = Image.objects.filter(user = request.user)
        result = []
        for img in images:
            create_time = img.create_time.strftime("%Y-%m-%d %H:%M:%S")
            result.append({
                "name":img.name,
                "url":request.scheme+"://"+request.META["HTTP_HOST"]+img.url,
                "id":img.pk,
                "create_time":create_time,
            })
        return Response(
            status=status.HTTP_200_OK,
            data={"images":result}
        )

    # 用户上传自定义图片
    @login_required
    def put(self, request):
        file = request.data["file"]
        target_img = PIL.Image.open(file)
        new_record = Image(user=request.user, name=request.data["name"], type="custom")
        filename = new_record.id + ".png"
        target_img.save("./media/"+filename)
        new_record.save()
        create_time = new_record.create_time.strftime("%Y-%m-%d %H:%M:%S")
        return Response(
            status=status.HTTP_200_OK,
            data={
                "url":request.scheme+"://"+request.META["HTTP_HOST"]+new_record.url,
                "id" :new_record.pk,
                "create_time": create_time,
                "name": new_record.name
            }
        )


class ImageManagementView(APIView):
    http_method_names = ["put","get","delete"]
    @login_required
    def put(self, request, pk):
        image = get_object_or_404(Image, pk=pk)
        image.name = request.name
        image.save()
        return Response(
            status=status.HTTP_200_OK,
            data={
                "message": "修改成功"
            }
        )

    @login_required
    def get(self, request, pk):
        record = get_object_or_404(Image,pk=pk)
        return Response(
            status=status.HTTP_200_OK,
            data={
                "name":record.name,
                "url":request.scheme+"://"+request.META["HTTP_HOST"]+record.url,
                "id":record.pk,
                "create_time":record.create_time
            }
        )

    @login_required
    def delete(self, request, pk):
        record = get_object_or_404(Image, pk=pk)
        record.delete()
        return Response(
            status=status.HTTP_200_OK,
            data={"message": "删除成功"}
        )