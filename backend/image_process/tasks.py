from __future__ import absolute_import, unicode_literals
from cmath import log
import logging
from backend import celery_app
import time
from backend.util import MapImageHelper
from backend.settings import predictor as P
from django.http import request,JsonResponse
from rest_framework import status
from image_process.models import Image, Task, Project
import PIL
from backend.Config import Config
from celery.utils.log import get_task_logger
import os
logger = get_task_logger(__name__)

def retrieval(task):
    if task.coordinate is None:
        task.imageA = task.project.imageA
    img_url = task.imageA
    img_a = PIL.Image.open(img_url)
    result_image, ratio = P.retrieval_predict(img_a)
    filename = str(task.id) + ".png"
    # task.analysis = {"retrieval": ratio}
    task.mask = "./media/"+filename
    task.status = "finished"
    result_image.save('./media/'+filename)
    task.save()

def sort(task):
    pass

def detection(task):
    pass

def contrast(task):
    pass


handle_func = {
    "retrieval":retrieval,
    "sort":sort,
    "detection":detection,
    "contrast":contrast
}


@celery_app.task(bind=True)
def image_handler(self, task_id):
    task = Task.objects.get(pk=task_id)
    type = task.project.type
    handle_func[type](task)

