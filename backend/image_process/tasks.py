from __future__ import absolute_import, unicode_literals
from cmath import log
import logging
from unicodedata import name
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
    imageA = task.project.imageA
    filename = imageA.url.split("/")[-1]
    img_a = PIL.Image.open(os.path.join("./media/", filename))
    if task.coordinate:
        tl = task.coordinate["tl"]
        br = task.coordinate["br"] 
        crop_img = img_a.crop((tl[0], tl[1], br[0], br[1]))
        H, W = crop_img.size
        new_record = Image(user = task.user, H = H, W = W, type="task", name="")
        filename = new_record.id + ".png"
        crop_img.save(os.path.join("./media/", filename))
        new_record.url = "/images/" + filename
        new_record.save()
        imageA = new_record
        img_a = crop_img
    task.imageA = imageA
    result_image, ratio = P.retrieval_predict(img_a)
    H, W = result_image.size
    mask = Image(user=task.user, H = H, W= W, type="mask", name="")
    filename = mask.id + ".png"
    result_image.save('./media/'+filename)
    mask.url = "/images/" + filename
    mask.save()
    # task.analysis = {"retrieval": ratio}
    task.mask = mask
    task.status = "finished"
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

