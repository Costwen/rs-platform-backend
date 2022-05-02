FROM redis:latest

RUN mkdir /etc/redis
ADD redis.conf /etc/redis/
#关闭daemon，前台执行
CMD ["redis-server", "/etc/redis/redis.conf;"]