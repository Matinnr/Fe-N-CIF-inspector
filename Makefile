# Convenience targets. All commands assume the CCDC-bundled Python.
#
# Configure the interpreter once via the environment variable:
#   export FEN_PY=~/CCDC/ccdc-software/csd-python-api/miniconda/bin/python
# or override per-call:
#   make test FEN_PY=/path/to/python

FEN_PY ?= $(HOME)/CCDC/ccdc-software/csd-python-api/miniconda/bin/python
APP    := app.py
PORT   ?= 8501

.PHONY: help demo test coverage lint smoke install clean

help:
	@echo "Targets:"
	@echo "  make demo       Launch the Streamlit app on port $(PORT)."
	@echo "  make test       Run the full pytest suite (no coverage)."
	@echo "  make coverage   Run pytest with line-by-line coverage on src/."
	@echo "  make smoke      Headless Streamlit launch + 5 s readiness check."
	@echo "  make install    Install pip dependencies into the CCDC env."
	@echo "  make clean      Remove __pycache__ / .pytest_cache."

demo:
	$(FEN_PY) -m streamlit run $(APP) --server.port $(PORT)

test:
	$(FEN_PY) -m pytest tests/

coverage:
	$(FEN_PY) -m pytest tests/ \
		--cov=src \
		--cov-report=term-missing \
		--cov-report=html:htmlcov

smoke:
	$(FEN_PY) -m streamlit run $(APP) --server.headless true \
		--server.port $(PORT) > /tmp/sl_smoke.log 2>&1 & \
	SL_PID=$$!; sleep 5; \
	if grep -q "Local URL" /tmp/sl_smoke.log; then \
		echo "✓ app started"; \
	else \
		echo "✗ app failed to start; log:"; \
		cat /tmp/sl_smoke.log; \
		kill $$SL_PID 2>/dev/null; exit 1; \
	fi; \
	kill $$SL_PID 2>/dev/null

install:
	$(FEN_PY) -m pip install streamlit plotly pytest pytest-cov

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage
