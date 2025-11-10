install-kernel:
	uv run python -m ipykernel install --user --name=berlin-bicycle-theft --display-name="Berlin bicycle theft"

mypy:
	uv run mypy bicycle_theft/

black:
	black --check bicycle_theft/

black-fix:
	black bicycle_theft/

ruff:
	ruff check bicycle_theft/ --fix

lint:
	make mypy && make black && make ruff