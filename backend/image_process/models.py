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
    name = models.CharField(verbose_name="image name", default="", max_length=25)
    raw = models.ImageField(verbose_name="raw image", default="", upload_to="raw")
    result = models.FileField(verbose_name="prediction result", upload_to="result")
    task = models.CharField(max_length=10,verbose_name="task type")
    # 以npy格式存储mask结果，以json存储检测结果

    def __str__(self):
        return "a normal inference result"

    class Meta:
        verbose_name = "Users' image that has been processed"
        indexes = [
            models.Index(fields=["user_id","upload_time"],name="inference_index")
        ]



