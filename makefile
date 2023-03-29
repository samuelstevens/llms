fmt:
	fd -e py | xargs isort
	fd -e py | xargs black

lint:
	fd -e py | xargs ruff
