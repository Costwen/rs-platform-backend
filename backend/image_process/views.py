from django.http import request,JsonResponse
from rest_framework import status
from rest_framework.decorators import api_view
# from backend.settings import predictor as P
from backend.util import MapImageHelper
from django.contrib.auth.models import AnonymousUser
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required,permission_required
import PIL
from image_process.models import *


@api_view(["POST"])
# @login_required(redirect_field_name = "next",login_url=None)
@permission_required("image_process.view_inference",raise_exception=True)
def retrieval(request):
    print(request.META["REMOTE_ADDR"])
    if request.data["source"] == "upload":
        img_file = request.FILES["img"]
    else:
        x1,y1,x2,y2 = eval(request.data["x1"]),eval(request.data["y1"]),eval(request.data["x2"]),eval(request.data["y2"])
        x1 =
        img_file = MapImageHelper.getImage()
    s = PIL.Image.open(img_file)  # 图片大小检查在前端完成,
    if isinstance(request.user,AnonymousUser):
        new_inference = Inference(user_id = get_user_model().objects.get(pk=1),raw = img_file)
        # img_file.save()
    else:
        new_inference = Inference(user_id=request.user, raw=img_file)
        # img_file.save()
    new_inference.save()
    if request.data["task"] == "road":
        pass
    elif request.data["task"] == "building":
        pass
    elif request.data["task"] == "water":
        pass
    # img_file.save()
    # result_img,result_stat = P.retrieval_predict(s)
    # result_img.show()
    return JsonResponse({
        "code":status.HTTP_200_OK,
        "raw_image_url":request.scheme+"://"+request.META["HTTP_HOST"]+"/images/"+new_inference.raw.name,
        # "stats":list(result_stat)
    })
# TODO:编写图片的Storage类，保存路径信息需要隐藏，文件重命名需要解决


@api_view(["POST"])
def sort(request):
    print(request.META["REMOTE_ADDR"])
    img_file = request.FILES["img"]
    s = PIL.Image.open(img_file)  # 图片大小检查在前端完成,
    # TODO：为方便测试，这里暂不要求登录
    if isinstance(request.user, AnonymousUser):
        new_inference = Inference(user_id=get_user_model().objects.get(pk=1), raw=img_file)
        # img_file.save()
    else:
        new_inference = Inference(user_id=request.user, raw=img_file)
        # img_file.save()
    new_inference.save()
    # img_file.save()
    # result_img,result_stat = P.retrieval_predict(s)
    # result_img.show()
    return JsonResponse({
        "code": status.HTTP_200_OK,
        "raw_image_url": request.scheme + "://" + request.META["HTTP_HOST"] + "/images/" + new_inference.raw.name,
        # "stats":list(result_stat)
    })


@api_view(["POST"])
def detection(request):
    return JsonResponse({
        "code": status.HTTP_200_OK
    })


@api_view(["POST"])
def contrast(request):
    return JsonResponse({
        "code": status.HTTP_200_OK
    })