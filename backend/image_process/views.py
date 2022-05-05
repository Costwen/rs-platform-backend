import time

from django.http import request,JsonResponse
from rest_framework import status
from rest_framework.decorators import api_view
from backend.settings import predictor as P
from backend.util import MapImageHelper
from django.contrib.auth.models import AnonymousUser
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required,permission_required
import PIL
from image_process.models import *
from django.shortcuts import get_object_or_404
from backend.Config import Config



def retrieval(request):
    print(request.META["REMOTE_ADDR"])
    if request.data["source"] == "upload":
        img_file = request.FILES["img"]
    else:
        x1,y1,x2,y2 = eval(request.data["x1"]),eval(request.data["y1"]),eval(request.data["x2"]),eval(request.data["y2"])
        img_file = MapImageHelper.getImage(x1,y1,x2,y2)
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


def sort(request):
    pass
    # return JsonResponse({
    #     "code": status.HTTP_200_OK,
    #     "raw_image_url": request.scheme + "://" + request.META["HTTP_HOST"] + "/images/" + new_inference.raw.name,
    #     # "stats":list(result_stat)
    # })


def detection(request):
    return JsonResponse({
        "code": status.HTTP_200_OK
    })


def contrast(request):
    return JsonResponse({
        "code": status.HTTP_200_OK
    })


# @login_required(redirect_field_name= "get_one_history",login_url=None)
@api_view(["GET"])
def get_specific_history(request):
    return JsonResponse({
        "code" : status.HTTP_200_OK
    })


# todo：缩略图！
# @login_required(redirect_field_name= "get_all_history",login_url=None)
@api_view(["GET"])
def get_all_history(request):
    user = request.user
    page_length = request.data["length"]
    inferences = user.inference_set.all()
    results = []
    for inference in inferences:
        results.append({
            "raw_image_url": request.scheme + "://" + request.META["HTTP_HOST"] + "/images/" + inference.raw.name,
            "upload_time": inference.upload_time,
            "task": inference.task,
            "name": inference.name,
            "id": inference.pk
        })
    return JsonResponse({
        "code": status.HTTP_200_OK,
        "results": results
    })


# TODO： account not done
@api_view(["PUT"])
def create_new_task(request):
    if request.data["mode"] == "openlayer":
        x1 = eval(request.data["tl_x"])
        y1 = eval(request.data["tl_y"])
        x2 = eval(request.data["br_x"])
        y2 = eval(request.data["br_y"])
        img_a = MapImageHelper.getImage(x1, y1, x2, y2)
    else:
        img_tmp = request.data["imageA"]
        img_a = PIL.Image.open(img_tmp).convert("RGB")
    create_time = time.time()
    if request.data['type'] == "retrieval":
        result_image, ratio = P.retrieval_predict(img_a)
        result_image.convert("RGB").save("./media/1.png")
        interval_time = time.time()-create_time
        return JsonResponse({
            "code": status.HTTP_200_OK,
            "mask": request.scheme + "://" + request.META["HTTP_HOST"] + "/images/" + "1.png",
            "result": [{
                "name": request.data["retrieval_type"],
                "ratio": ratio.item(0)
            }],
            "inference_time": str(interval_time)+"s"
        })
    elif request.data["type"] == "sort":
        result_image, mask_bincount = P.sort_predict(img_a)
        result_image.convert("RGB").save("./media/1.png")
        interval_time = time.time() - create_time
        return JsonResponse({
            "code": status.HTTP_200_OK,
            "mask": request.scheme + "://" + request.META["HTTP_HOST"] + "/images/" + "1.png",
            "result": [{
                "name": Config.sort_category[i],
                "ratio": mask_bincount.item(i)
            } for i in range(0, len(Config.sort_category))],
            "inference_time": str(interval_time)+"s"
        })
    elif request.data["type"] == "contrast":
        pass
    elif request.data["type"] == "detection":
        pass




@login_required(redirect_field_name= "change_task_info",login_url=None)
@api_view(["POST","DEL"])
def change_task_info(request):
    pass


@login_required()
@api_view(["GET"])
def result_detail(request):
    user = request.user
    result = get_object_or_404(Inference,user = user,pk = request.GET["id"])
    return JsonResponse