ALTER user 'dbuser'@'%' identified with mysql_native_password BY 'PASSWORD';
GRANT ALL PRIVILEGES ON remote_sensing.* TO 'dbuser'@'%';
FLUSH PRIVILEGES;