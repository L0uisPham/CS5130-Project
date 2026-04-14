.PHONY: run-api

run-api:
	python -m uvicorn api.app:app --reload --host 0.0.0.0 --port 8001
