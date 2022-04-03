from django.shortcuts import render
import os
from django.http import request,JsonResponse
from rest_framework import status
from rest_framework.decorators import api_view
from backend.settings import predictor as P
import PIL
# Create your views here.

@api_view(["POST"])
def retrieval(request):
    img_file = request.FILES["img"]
    s = PIL.Image.open(img_file)
    # img_file.save()
    result = P.retrieval_predict(s)
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