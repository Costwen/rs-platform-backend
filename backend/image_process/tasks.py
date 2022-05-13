from __future__ import absolute_import, unicode_literals
from backend import celery_app
import time
from backend.util import MapImageHelper
from backend.settings import predictor as P
from django.http import request,JsonResponse
from rest_framework import status
from image_process.models import Image, Task, Project
import PIL
from backend.Config import Config
def retrieval(task):
    result_image, ratio = P.retrieval_predict(img_a)
    pass

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
def image_handler(self, project_id, user_id, coordinate):
    task_id = self.request.id
    task = Task.objects.get(pk=task_id)
    handle_func[type](task)

