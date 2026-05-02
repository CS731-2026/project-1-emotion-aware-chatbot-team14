# ── Experiments Makefile ──────────────────────────────────────────────────────
# Manages virtual environment for experiments
#
# Targets:
#   make init      Initialize venv and install dependencies
#   make activate  Show activation command
#   make clean     Remove venv and build artifacts

VENV       := venv
PYTHON     := $(VENV)/bin/python
PIP        := $(PYTHON) -m pip

.PHONY: init activate clean

$(VENV):
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

init: $(VENV)
	@echo "Checking for requirements.txt files in subdirectories..."
	@if [ -f emotion_recognition/requirements.txt ]; then \
		echo "Installing emotion_recognition requirements..."; \
		$(PIP) install -r emotion_recognition/requirements.txt; \
	fi
	@if [ -f emotion_recognition/learning/requirements.txt ]; then \
		echo "Installing emotion_recognition/learning requirements..."; \
		$(PIP) install -r emotion_recognition/learning/requirements.txt; \
	fi
	@echo "✓ Experiments environment initialized"
	@echo ""
	@echo "To activate the environment, run:"
	@echo "  source venv/bin/activate"

activate:
	bash --init-file activate.sh -i

clean:
	rm -rf $(VENV)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✓ Cleaned experiments environment"
