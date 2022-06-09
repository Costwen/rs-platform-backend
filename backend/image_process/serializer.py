from dataclasses import fields
from venv import create
from rest_framework import serializers
from .models import Image

class ImageSerializer(serializers.ModelSerializer):
    create_time = serializers.DateTimeField(format="%Y-%m-%d %H:%M:%S",read_only=True)
    url = serializers.SerializerMethodField()
    def get_url(self, obj):
        return "+"+ obj.url
    class Meta:
        model = Image
        exclude = ("user",)