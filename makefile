add-uv:
	@echo "--- 🚀 Installing UV ---"	
	curl -LsSf https://astral.sh/uv/install.sh | sh
	# windows:
	# powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

install:
	@echo "--- 🚀 Installing project ---"
	uv sync --extra dev --extra docs --extra tests

lint:
	@echo "--- 🧹 Running linters ---"
	uv run ruff format . 						            # running ruff formatting
	uv run ruff check **/*.py --fix						# running ruff linting

lint-check:
	@echo "--- 🧹 Check is project is linted ---"
	uv run ruff format . --check						    # running ruff formatting
	uv run ruff check **/*.py 						        # running ruff linting

