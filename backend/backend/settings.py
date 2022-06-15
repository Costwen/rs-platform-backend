"""
Django settings for backend project.

Generated by 'django-admin startproject' using Django 3.1.13.

For more information on this file, see
https://docs.djangoproject.com/en/3.1/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/3.1/ref/settings/
"""

from pathlib import Path
from backend.util import Predictor
from backend.Config import Config
import os

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/3.1/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'xptv%pmrnbvf@f_i89qv9zux#ad^ihoqq0ll+8!^+%yk2779lr'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ["*"]

AUTH_USER_MODEL = 'account.User'

AUTHENTICATION_BACKENDS = [
    "django.contrib.auth.backends.ModelBackend",
    'account.auth_backend.Email_auth'
]

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    "image_process",
    "account",
    "rest_framework",
    "channels"
]

#TODO:生产环境下请把CSRF打开。

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    # 'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'backend.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'backend.wsgi.application'

# Database
# https://docs.djangoproject.com/en/3.1/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'OPTIONS':{
            "init_command":"SET sql_mode='STRICT_TRANS_TABLES'"
        },
        'NAME': 'remote_sensing',
        'USER': 'backend',
        'PASSWORD': 'buaa2022gogogo',
        'HOST': '121.37.88.255',
        'PORT': '3306',
        'CONN_MAX_AGE': 60
    }
}

REST_FRAMEWORK = {

    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    )

}

# Password validation
# https://docs.djangoproject.com/en/3.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
# https://docs.djangoproject.com/en/3.1/topics/i18n/


 # 设置ASGI应用
ASGI_APPLICATION = "backend.asgi.application"

CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            "hosts": ["redis://:jxdhandsome@101.43.134.156:6378/2"],
        },
    },
}

# redis配置
CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": "redis://:jxdhandsome@101.43.134.156:6378/3",
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
            "CONNECTION_POOL_KWARGS": {"max_connections": 100}
        }
    }
}


LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/3.1/howto/static-files/

STATIC_URL = '/static/'

MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

SESSION_COOKIE_AGE = 360000
predictor = Predictor(Config)

# celery相关配置
# 配置celery时区，默认时UTC。
timezone = TIME_ZONE

# 任务队列的链接地址 celery配置redis作为broker。redis有16个数据库，编号0~15，这里使用第1个。
# CELERY_result_backend = 'redis://buaa2022gogogo@101.43.134.156:6379/1'

CELERY_BROKER_URL = 'redis://:jxdhandsome@101.43.134.156:6378/0' # Broker配置，使用Redis作为消息中间件

CELERY_RESULT_BACKEND = 'redis://:jxdhandsome@101.43.134.156:6378/1' # BACKEND配置，这里使用redis

CELERY_RESULT_EXPIRES = None  # 存储结果过期时间

CELERY_RESULT_SERIALIZER = 'json' # 结果序列化方案


# 可选参数：给某个任务限流
# task_annotations = {'tasks.my_task': {'rate_limit': '10/s'}}

# 可选参数：给任务设置超时时间。超时立即中止worker
# task_time_limit = 10 * 60

# 可选参数：给任务设置软超时时间，超时抛出Exception
# task_soft_time_limit = 10 * 60

# 可选参数：如果使用django_celery_beat进行定时任务
# beat_scheduler = "django_celery_beat.schedulers:DatabaseScheduler"

# 更多选项见
# https://docs.celeryproject.org/en/stable/userguide/configuration.html

from datetime import timedelta

SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(days=1),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=10),
}