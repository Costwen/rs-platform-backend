ALTER user 'backend'@'%' identified with mysql_native_password BY 'buaa2022gogogo';
GRANT ALL PRIVILEGES ON remote_sensing.* TO 'backend'@'%';
FLUSH PRIVILEGES;