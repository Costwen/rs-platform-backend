from django.contrib import admin
# Register your models here.


class InferenceAdmin(admin.ModelAdmin):
    date_hierarchy = "upload_time"
