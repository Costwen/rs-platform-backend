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
#celery -A backend worker -l info&&
/etc/init.d/celeryd start
daphne -p 9000 -b 0.0.0.0 backend.asgi:application
# tail空命令，保证有一个任务在前台执行，防止容器退出
tail -f /root/backend/requirements.txt
# 若用户在启动容器时加入了其他命令，那么将会在这里执行，提高拓展性。
exec "$@"
