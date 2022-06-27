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
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from django.core.cache import cache

channel_layer = get_channel_layer()

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
    message = {
        'type': 'websocket.celery',
        "message": {
            "id": str(task.id),
            "status": task.status,
        }
    }
    print(cache.get(task.user.pk))
    async_to_sync(channel_layer.send)(
        cache.get(task.user.pk),
        message
    )

def sort(task):
    pass

def detection(task):
    pass

def contrast(task):
    imageA,imageB = task.project.imageA,task.project.imageB
    filename_a,filename_b = imageA.url.split("/")[-1],imageB.url.split("/")[-1]
    img_a,img_b = PIL.Image.open(os.path.join("./media/", filename_a)), PIL.Image.open(os.path.join("./media/",filename_b))
    if task.coordinate:
        tl = task.coordinate["tl"]
        br = task.coordinate["br"]
        crop_a, crop_b = img_a.crop((tl[0], tl[1], br[0], br[1])), img_b.crop((tl[0],tl[1],br[0],br[1]))
        H, W = crop_a.size
        assert crop_a.size == crop_b.size
        new_record_a, new_record_b = Image(user=task.user, H=H, W=W, type="task", name=""),Image(user=task.user, H=H, W=W, type="task", name="")
        filename_a, filename_b = new_record_a.id + ".png", new_record_b.id + ".png"
        crop_a.save(os.path.join("./media/", filename_a))
        crop_b.save(os.path.join("./media/", filename_b))
        new_record_a.url, new_record_b.url = "/images/" + filename_a, "/images/" + filename_b
        new_record_a.save()
        new_record_b.save()
        imageA,imageB = new_record_a, new_record_b
        img_a,img_b = crop_a, crop_b
    task.imageA, task.imageB = imageA,imageB
    result_img,ratio = P.contrast_predict(img_a, img_b)
    H, W = result_img.size
    mask = Image(user=task.user, H=H, W=W, type="mask", name="")
    filename = mask.id + ".png"
    result_img.save('./media/' + filename)
    mask.url = "/images/" + filename
    mask.save()
    task.mask = mask
    task.status = "finished"
    task.save()
    message = {
        'type': 'websocket.celery',
        "message": {
            "id": str(task.id),
            "status": task.status,
        }
    }
    print(cache.get(task.user.pk))
    async_to_sync(channel_layer.send)(
        cache.get(task.user.pk),
        message
    )


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

