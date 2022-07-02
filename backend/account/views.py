import email
from rest_framework import status
from rest_framework.decorators import *
from account.models import *
from rest_framework.response import Response
from rest_framework.decorators import api_view

@api_view(["POST"])
def register_view(request):
    try:
        username = request.data["user"]
        password = request.data["pass"]
    except:
        return Response({"message":"username or password is missing"},status=status.HTTP_400_BAD_REQUEST)
    print(username,password)
    User.objects.create_user(username = username,password=password, email=None)
    return Response({"message":"user created"},status=status.HTTP_200_OK)
