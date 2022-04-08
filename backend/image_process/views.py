from django.http import request,JsonResponse
from rest_framework import status
from rest_framework.decorators import api_view
# from backend.settings import predictor as P
from django.contrib.auth.decorators import login_required,permission_required
import PIL
from image_process.models import *

@api_view(["POST"])
@login_required(redirect_field_name = "next",login_url=None)
@permission_required("image_process.view_inference",raise_exception=True)
def retrieval(request):
    print(request.META["REMOTE_ADDR"])
    img_file = request.FILES["img"]
    print(img_file)
    s = PIL.Image.open(img_file)
    # img_file.save()
    # result = P.retrieval_predict(s)
    return JsonResponse({
        "code":status.HTTP_200_OK
    })


@api_view(["POST"])
def sort(request):
    return JsonResponse({
        "code": status.HTTP_200_OK
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