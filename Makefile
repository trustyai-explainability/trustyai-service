# Developer entry points. `uv` installs only the `dev` group, so the flags
# below are required: `uv run pytest` alone fails with "unrecognized arguments:
# --dist=loadgroup -n" from the pyproject addopts.

# Required everywhere: pyrefly type-checks tests that import pytest_asyncio,
# which only the test group provides.
GROUPS := --all-groups

# Backends for the ungated suite. psycopg is excluded so the PostgreSQL storage
# tests skip through importorskip on a machine with no server; the gated targets
# below add it. Listed rather than --all-extras so a future extra cannot join
# local runs.
EXTRAS := --extra mariadb --extra sqlite --extra migration --extra mmd
DB_EXTRAS := $(EXTRAS) --extra postgres

# Credentials match the public CI service containers.
MIGRATION_MARIADB_URL := mysql+asyncmy://trustyai:trustyai@127.0.0.1:3306/trustyai-database  # pragma: allowlist secret
MIGRATION_POSTGRES_URL := postgresql+asyncpg://trustyai:trustyai@127.0.0.1:5432/trustyai-database  # pragma: allowlist secret

# Local containers only, no cluster; a missing server fails instead of skipping.
DB_TEST_ENV := TRUSTYAI_REQUIRE_LIVE_SQL_TESTS=1 \
               TRUSTYAI_MIGRATION_TEST_MARIADB_URL=$(MIGRATION_MARIADB_URL) \
               TRUSTYAI_MIGRATION_TEST_POSTGRES_URL=$(MIGRATION_POSTGRES_URL)

.PHONY: help sync test test-databases migration-test lint bandit typecheck check
.PHONY: requirements requirements-build deps migrate

help:
	@printf '%s\n' \
		'sync                install every dependency group plus the CI extras' \
		'test                run the suite; live SQL tests skip without databases' \
		'test-databases      run the suite with live SQL gates (needs local databases)' \
		'migration-test      run the serial migration suite (needs local databases)' \
		'lint                ruff check and ruff format --check' \
		'bandit              bandit security scan' \
		'typecheck           pyrefly' \
		'check               lint + bandit + typecheck' \
		'requirements        regenerate hashed runtime requirements from uv.lock' \
		'requirements-build  recompile hashed build requirements (bumps unpinned deps)' \
		'deps                uv lock --upgrade, then regenerate both requirements files' \
		'migrate             run trustyai-migrate with ARGS="..."'

sync:
	uv sync $(GROUPS) $(EXTRAS)

# CI splits the migration suite out and runs it serially: those tests reset
# tables shared with the rest of the schema.
test:
	uv sync $(GROUPS) $(EXTRAS)
	uv run $(GROUPS) $(EXTRAS) pytest tests/ -v --ignore=tests/service/data/migrations
	uv run $(GROUPS) $(EXTRAS) pytest tests/service/data/migrations -v -n 0

# CI test job's first phase; needs both databases on 127.0.0.1. No -n: addopts
# sets `-n 4 --dist=loadgroup`, and xdist_group keeps each database serial.
test-databases:
	$(DB_TEST_ENV) uv run $(GROUPS) $(DB_EXTRAS) pytest tests/ -v --ignore=tests/service/data/migrations

# Serial: migration tests share database state, so -n 0 overrides addopts.
migration-test:
	$(DB_TEST_ENV) uv run $(GROUPS) $(DB_EXTRAS) pytest tests/service/data/migrations -v -n 0

lint:
	uv run $(GROUPS) ruff check src tests
	uv run $(GROUPS) ruff format --check src tests

bandit:
	uv run $(GROUPS) bandit -c pyproject.toml -r src

typecheck:
	uv run $(GROUPS) pyrefly check

check: lint bandit typecheck

# --frozen exports the current lock; run `uv lock` first if pyproject changed.
requirements:
	uv export --frozen --format requirements.txt --extra mariadb --extra postgres --extra migration --extra mmd --no-dev --no-emit-project --emit-index-url --output-file requirements.txt

# Unpinned inputs, so a run can bump build backends. Review the diff and rebuild
# the container, which installs these files with --require-hashes.
requirements-build:
	uv pip compile requirements-build.in --python-version 3.12 --python-platform linux --emit-index-url --emit-index-annotation --generate-hashes --prerelease=if-necessary --no-strip-markers -o requirements-build.txt --default-index https://pypi.org/simple

deps:
	uv lock --upgrade
	$(MAKE) requirements
	$(MAKE) requirements-build

# Pass flags through; the CLI validates source and target. For example:
#   make migrate ARGS="--source pvc --source-path /data --quiesced \
#     --target-url postgresql+asyncpg://127.0.0.1:5432/trustyai-database"
migrate:
	uv run $(GROUPS) $(EXTRAS) trustyai-migrate $(ARGS)
