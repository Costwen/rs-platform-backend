from django.db import models
from backend import settings
from django.contrib.auth.models import AbstractUser
# Create your models here.
import uuid


def uuid_str():
    id = uuid.uuid4()
    return str(id)

class Image(models.Model):
    user = models.ForeignKey(
        to=settings.AUTH_USER_MODEL,
        related_name="image_list",
        verbose_name="user that own the image",
        on_delete=models.CASCADE
    )
    id = models.UUIDField(verbose_name="image id", primary_key=True, default=uuid_str, editable=False)
    url = models.URLField(verbose_name="image url", default="", max_length=1024)
    H = models.IntegerField(verbose_name="image height", default=0)
    W = models.IntegerField(verbose_name="image width", default=0)
    name = models.CharField(verbose_name="image name", default="未命名", max_length=100)
    create_time = models.DateTimeField(auto_now_add=True)
    type = models.CharField(verbose_name="image type", default="", max_length=100)
    def __str__(self):
        return "a image"

class Project(models.Model):
    user = models.ForeignKey(
        to=settings.AUTH_USER_MODEL,
        related_name="project_list",
        verbose_name="user that own the project",
        on_delete=models.CASCADE
    )
    name = models.CharField(verbose_name="project name", max_length=25)
    id = models.UUIDField(verbose_name="project id", primary_key=True, default=uuid_str, editable=False)
    imageA = models.ForeignKey(to=Image, related_name="imageA_project", verbose_name="imageA", on_delete=models.DO_NOTHING, blank=True, null=True)
    imageB = models.ForeignKey(to=Image, related_name="imageB_project", verbose_name="imageB", on_delete=models.DO_NOTHING, blank=True, null=True)
    
    create_time = models.DateTimeField(auto_now_add=True, verbose_name="create time")
    modify_time = models.DateTimeField(auto_now=True, verbose_name="modify time")
    status = models.CharField(verbose_name="project status", max_length=1024, default="normal")  # normal, deleted
    type = models.CharField(verbose_name="project type", max_length=100, default="")
    


class Task(models.Model):
    user = models.ForeignKey(
        to=settings.AUTH_USER_MODEL,
        related_name="task_list",
        verbose_name="user that own the task",
        on_delete=models.CASCADE
    )
    project = models.ForeignKey(
        to=Project,
        related_name="task_list",
        verbose_name="project that the task belongs to",
        on_delete=models.CASCADE
    )
    id = models.UUIDField(verbose_name="task id", primary_key=True, default=uuid_str, editable=False)
    status = models.CharField(verbose_name="task status", max_length=10, default="pending")
    create_time = models.DateTimeField(auto_now_add=True)
    
    imageA = models.ForeignKey(to=Image, related_name="imageA_task", verbose_name="inputA", on_delete=models.CASCADE, blank=True, null=True)
    imageB = models.ForeignKey(to=Image, related_name="imageB_task", verbose_name="inputB", on_delete=models.CASCADE, blank=True, null=True)
    mask = models.ForeignKey(to=Image, related_name="mask_task", verbose_name="mask", on_delete=models.CASCADE, blank=True, null=True)
    coordinate = models.JSONField(verbose_name="coordinate result", default=dict, blank=True, null=True)
    analysis = models.JSONField(verbose_name="analysis result", default=dict)
    
    def __str__(self):
        return "a task"




