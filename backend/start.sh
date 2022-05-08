#!/bin/bash
# 先等MySQL服务启动
#while ! nc -z db 3306 ;
#do
#  echo "Waiting for the MySQL Server"
#  sleep 3
#done

python manage.py makemigrations account image_process&&
python manage.py migrate&&
uwsgi /root/backend/uwsgi.ini&&
# tail空命令，保证有一个任务在前台执行，防止容器退出
tail -f /dev/null
# 若用户在启动容器时加入了其他命令，那么将会在这里执行，提高拓展性。
exec "$@"
