import json
from rest_framework import status
from sympy import re
from backend.settings import predictor as P
from backend.util import MapImageHelper
from django.contrib.auth.models import AnonymousUser
from django.contrib.auth import get_user_model
import PIL
from image_process.models import *
from django.shortcuts import get_object_or_404
from image_process.tasks import *
from image_process.tasks import image_handler
from rest_framework.response import Response
from rest_framework.views import APIView
import demjson
from .serializer import ImageSerializer
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

def image2json(image):
    serizalizer = ImageSerializer(instance=image)
    return serizalizer.data


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
                "mask": image2json(task.mask),
                "imageA": image2json(task.imageA),
                "imageB": image2json(task.imageB),
                "create_time":task.create_time.strftime("%Y-%m-%d %H:%M:%S"),
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
            data={
                "message":"获取成功",
                "task":{
                    "id":task.pk,
                    "project_id":task.project_id,
                    "coordinate":task.coordinate,
                    "status":task.status,
                    "mask": image2json(task.mask),
                    "imageA": image2json(task.imageA),
                    "imageB": image2json(task.imageB),
                    "create_time":task.create_time.strftime("%Y-%m-%d %H:%M:%S"),
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
                "imageA": image2json(project.imageA),
                "imageB": image2json(project.imageB),
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
                "mask": image2json(task.mask),
                "imageA": image2json(task.imageA),
                "imageB": image2json(task.imageB),
                "create_time":task.create_time.strftime("%Y-%m-%d %H:%M:%S"),
            })
        data = {
            "name": project.name,
            "type": project.type,
            "imageA": image2json(project.imageA),
            "imageB": image2json(project.imageB),
            "id": project.pk,
            "create_time": project.create_time.strftime("%Y-%m-%d %H:%M:%S"),
            "modify_time": project.modify_time.strftime("%Y-%m-%d %H:%M:%S"),
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
        new_record = Image(user = request.user, name=name, type = "public")
        coordinate = json.loads(coordinate)
        target_img = MapImageHelper.getImage(coordinate["tl"][0],coordinate["br"][1],coordinate["br"][0],coordinate["tl"][1])
        filename = new_record.id + ".png"
        H, W = target_img.size
        new_record.H = H
        new_record.W = W
        target_img.save("./media/"+filename)
        new_record.url = "/images/" + filename
        new_record.save()
        return Response(
            status=status.HTTP_200_OK,
            data={
                "message": "上传成功",
                "image": image2json(new_record),
            }
        )

    # 用户获得自己的所有图片
    @login_required
    def get(self, request):
        images = Image.objects.filter(user = request.user)
        result = []
        for img in images:
            result.append(image2json(img))
        return Response(
            status=status.HTTP_200_OK,
            data={
                "images": result,
            }
        )

    # 用户上传自定义图片
    @login_required
    def put(self, request):
        file = request.data["file"]
        target_img = PIL.Image.open(file)
        H, W = target_img.size
        new_record = Image(user=request.user, name=request.data["name"], type="custom", H=H, W=W)
        filename = new_record.id + ".png"
        target_img.save("./media/"+filename)
        new_record.save()
        return Response(
            status=status.HTTP_200_OK,
            data={
                "message": "上传成功",
                "image": image2json(new_record),
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
                "image": image2json(record),
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