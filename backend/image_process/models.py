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
    # detection = models.ImageField(verbose_name="object-detection result of image",upload_to="/detection")
    # 似乎不能仅以图像形式存储结果？


    def __str__(self):
        return "a normal inference result"

    class Meta:
        verbose_name = "Users' image that has been processed"



