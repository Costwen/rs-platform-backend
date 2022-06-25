from dataclasses import fields
from venv import create
from rest_framework import serializers
from .models import Image, Project

class ImageSerializer(serializers.ModelSerializer):
    create_time = serializers.DateTimeField(format="%Y-%m-%d %H:%M:%S",read_only=True)
    url = serializers.SerializerMethodField()
    project = serializers.SerializerMethodField()
    def get_url(self, obj):
        return "http://101.43.134.156"+ obj.url
    def get_project(self, obj):
        type = obj.type
        if type != "mask":
            return ''
        task = obj.mask_task.get()
        return task.project.pk
    class Meta:
        model = Image
        exclude = ("user",)