from django.db import models
from django.contrib.auth.models import AbstractBaseUser,BaseUserManager,PermissionsMixin
from django.contrib.auth.validators import UnicodeUsernameValidator
# Create your models here.


class MyUserManager(BaseUserManager):
    def _create_user(self, email, username, password, **kwargs):
        email = self.normalize_email(email)
        username = self.model.normalize_username(username)
        user = self.model(email=email, username=username, **kwargs)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_user(self,email,username,password,**kwargs):
        return self._create_user(email,username,password,**kwargs)

    def create_superuser(self,email,username,password,**kwargs):
        return self._create_user(email,username,password,is_admin=True,**kwargs)


class User(AbstractBaseUser,PermissionsMixin):
    username_validator = UnicodeUsernameValidator()
    username = models.CharField(
        max_length=150,
        unique=True,
        help_text="Required. 150 characters or fewer. Letters, digits and @/./+/-/_ only.",
        validators=[username_validator],
        error_messages = {
            'unique': "A user with that username already exists.",
        },
    )
    is_admin = models.BooleanField("is admin",default = False)
    email = models.EmailField("email address",unique=True,blank = False)
    date_joined = models.DateTimeField("date joined",auto_now_add=True)
    avatar = models.ImageField(verbose_name="user_avatar",upload_to="avatars",default="default_avatar.jpg")  # 可以规定头像的大小

    objects = MyUserManager()

    EMAIL_FIELD = 'email'
    USERNAME_FIELD = 'username'
    REQUIRED_FIELDS = ['email']

    class Meta:
        db_table = 'user'
        indexes = [
            models.Index(fields=['email',"password"],name="email_auth_index")
        ]
