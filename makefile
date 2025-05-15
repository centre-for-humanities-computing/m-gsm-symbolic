add-uv:
	@echo "--- 🚀 Installing UV ---"	
	curl -LsSf https://astral.sh/uv/install.sh | sh
	# windows:
	# powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

install:
	@echo "--- 🚀 Installing project ---"
	uv sync

lint:
	@echo "--- 🧹 Running linters ---"
	uv run ruff format . 						            # running ruff formatting
	uv run ruff check **/*.py --fix						# running ruff linting

lint-check:
	@echo "--- 🧹 Check is project is linted ---"
	uv run ruff format . --check						    # running ruff formatting
	uv run ruff check **/*.py 						        # running ruff linting

do-translation:
	@echo "--- Perform initial translation of the data ---"
	uv run src/scripts/initial_translation.py

do-post-training:
	@echo "--- Perform post training using GRPO ---"
	uv run src/scripts/post_training.py