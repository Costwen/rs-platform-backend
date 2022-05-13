from email.policy import default
from django.db import models
from matplotlib import projections
from backend import settings
from django.contrib.auth.models import AbstractUser
# Create your models here.
import uuid


def uuid_str():
    return str(uuid.uuid4())

class Project(models.Model):
    user = models.ForeignKey(
        to=settings.AUTH_USER_MODEL,
        related_name="project_user",
        verbose_name="user that own the project",
        on_delete=models.CASCADE
    )
    name = models.CharField(verbose_name="project name", max_length=25)
    id = models.UUIDField(verbose_name="project id", primary_key=True, default=uuid_str(), editable=False)
    imageA = models.URLField(verbose_name="imageA url", default="", max_length=1024, blank=True)
    imageB = models.URLField(verbose_name="imageB url", default="", max_length=1024, blank=True)
    create_time = models.DateTimeField(auto_now_add=True)
    type = models.CharField(verbose_name="project type", max_length=100, default="")


class Task(models.Model):
    user = models.ForeignKey(
        to=settings.AUTH_USER_MODEL,
        related_name="task_user",
        verbose_name="user that own the task",
        on_delete=models.CASCADE
    )
    project = models.ForeignKey(
        to=Project,
        related_name="project",
        verbose_name="project that the task belongs to",
        on_delete=models.CASCADE
    )
    id = models.UUIDField(verbose_name="task id", primary_key=True, default=uuid_str(), editable=False)
    status = models.CharField(verbose_name="task status", max_length=10, default="pending")
    create_time = models.DateTimeField(auto_now_add=True)
    mask = models.URLField(verbose_name="mask url", default="", max_length=1024, blank=True)
    imageA = models.URLField(verbose_name="imageA url", default="", max_length=1024, blank=True)
    imageB = models.URLField(verbose_name="imageB url", default="", max_length=1024, blank=True)
    coordinate = models.JSONField(verbose_name="coordinate result", default=dict)
    analysis = models.JSONField(verbose_name="analysis result", default=dict)

    def __str__(self):
        return "a task"


class Image(models.Model):
    user = models.ForeignKey(
        to=settings.AUTH_USER_MODEL,
        related_name="image_user",
        verbose_name="user that own the image",
        on_delete=models.CASCADE
    )
    id = models.UUIDField(verbose_name="image id", primary_key=True, default=uuid_str(), editable=False)
    url = models.URLField(verbose_name="image url", default="", max_length=1024)
    name = models.CharField(verbose_name="image name", default="", max_length=100)
    create_time = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return "a image"