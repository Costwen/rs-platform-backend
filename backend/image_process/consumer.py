
from cgitb import text
import json
from channels.generic.websocket import WebsocketConsumer   #异步，实现更好的性能
from django.core.cache import cache
class ChatConsumer(WebsocketConsumer):
    def websocket_connect(self, message):
        """客户端请求建立链接时 自动触发"""
        self.accept()  # 建立链接  并且自动帮你维护每一个客户端
        cache.set(self.scope["user"].pk, self.channel_name)

    def websocket_receive(self, message):
        """客户端发送数据过来  自动触发"""
        print(message, '----------')
        message = message.get('text')
        # self.send(text_data=message)

    def websocket_celery(self, message):
        """celery发送数据过来  自动触发"""
        print(message, '----------')
        message = message.get('message')
        message = json.dumps(message)
        self.send(text_data=message)

    def websocket_disconnect(self, message):
        """客户端断开链接之后  自动触发"""
        cache.delete(self.scope["user"].pk)
        print('清除成功')
        self.close()