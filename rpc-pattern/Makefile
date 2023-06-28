run-locust:
	poetry run locust -f locust-test.py --host=http://127.0.0.1:8001

run-front-server:
	poetry run uvicorn frontend-server:app --reload --port 8001

run-back-server:
	WORKERS=100 poetry run python backend-server.py

format:
	poetry run isort .
	poetry run black .