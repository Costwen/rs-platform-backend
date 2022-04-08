from django.db import models
from backend import settings
from django.contrib.auth.models import AbstractUser
# Create your models here.

class Inference(models.Model):
    user_id = models.ForeignKey(
        to=settings.AUTH_USER_MODEL,
        related_name="user",
        verbose_name="user that possess the image",
        on_delete=models.CASCADE
    )
    upload_time = models.DateTimeField(auto_now_add=True)


    def __str__(self):
        return "a normal inference result"

    class Meta:
        verbose_name = "Users' image that has been processed"



