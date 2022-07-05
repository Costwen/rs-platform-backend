FROM nginx:latest
#删除原有配置，创建static文件夹和ssl文件夹
RUN rm /etc/nginx/conf.d/default.conf \
&& mkdir -p /usr/share/nginx/html/static \
&& mkdir -p /usr/share/nginx/html/media \
&& mkdir -p /usr/share/nginx/ssl \
&& rm /etc/nginx/nginx.conf
#设置文件夹的用户和用户组，否则会因为权限问题导致上传的图片无法正确显示
RUN chown -R root:root /usr/share/nginx/html/media \
&& chmod -R 775 /usr/share/nginx/html/media
ADD dist /etc/nginx/

ADD mysite.conf /etc/nginx/conf.d/
ADD nginx.conf /etc/nginx/
#关闭daemon，前台执行
RUN tail /etc/nginx/conf.d/mysite.conf
CMD ["nginx", "-g", "daemon off;"]
