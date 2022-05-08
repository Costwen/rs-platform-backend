from __future__ import absolute_import, unicode_literals
import os
from celery import Celery
from celery.schedules import crontab
from datetime import timedelta
from django.conf import settings

# 指定Django默认配置文件模块
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')

# 为我们的项目myproject创建一个Celery实例。这里不指定broker backend 容易出现错误。
# 如果没有密码 使用 'redis://127.0.0.1:6379/0'
app = Celery('celery')

# 这里指定从django的settings.py里读取celery配置
app.config_from_object('django.conf:settings', namespace="CELERY")

# 自动从所有已注册的django app中加载任务
app.autodiscover_tasks(lambda: settings.INSTALLED_APPS)