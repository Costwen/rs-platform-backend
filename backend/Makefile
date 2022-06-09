.PHONY: django celery
django:
	python manage.py runserver
celery:
	celery -A backend worker -l info

