FROM python:3.7

RUN mv /etc/apt/sources.list /etc/apt/sources.list.backup
COPY sources.list /etc/apt/sources.list
RUN apt-get update && apt-get install -y netcat

MAINTAINER jxd
# 设置解释器不要生成字节码 pyc
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get install -y python3-opencv

COPY pip.conf /root/.pip/pip.conf

RUN pip config set global.index-url http://mirrors.aliyun.com/pypi/simple
RUN pip config set install.trusted-host mirrors.aliyun.com
RUN pip uninstall protobuf
RUN pip install protobuf==3.19.0
RUN python -m pip install paddlepaddle==2.2.2 -i https://mirror.baidu.com/pypi/simple
ENV APP_HOME=/root/backend
RUN mkdir -p $APP_HOME
WORKDIR $APP_HOME

ADD . $APP_HOME

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY celeryd.conf /etc/default/celeryd
COPY celeryd.init /etc/init.d/celeryd
RUN chmod 0777 /etc/init.d/celeryd
#RUN python -m pip install paddlepaddle==2.2.2 -i https://mirror.baidu.com/pypi/simple

RUN chmod +x ./start.sh
# 指定容器运行后执行的命令，这里指定运行启动脚本
ENTRYPOINT  /bin/bash ./start.sh