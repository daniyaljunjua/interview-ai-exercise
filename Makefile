install:
	uv sync --all-extras

########################################################################################################################
# Quality checks
########################################################################################################################

test:
	uv run pytest tests

lint:
	uv run ruff check ai_exercise tests

format:
	uv run ruff check ai_exercise tests --fix

typecheck:
	uv run mypy ai_exercise


########################################################################################################################
# Api
########################################################################################################################

start-api:
	docker compose up

dev-api:
	uv run ai_exercise/main.py

########################################################################################################################
# Streamlit
########################################################################################################################

start-app:
	uv run streamlit run demo/main.py

########################################################################################################################
# Evaluation
########################################################################################################################

eval:
	@echo "Loading naive collection..."
	@curl -sf "http://localhost:80/load?strategy=naive" || (echo "Server not running. Start it with 'make dev-api' first." && exit 1)
	@echo "\nLoading structural collection..."
	@curl -sf "http://localhost:80/load?strategy=structural"
	@echo "\nRunning evaluation..."
	uv run python -m ai_exercise.eval.run naive structural

