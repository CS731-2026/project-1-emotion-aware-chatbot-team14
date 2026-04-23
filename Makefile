# ── Root Makefile ────────────────────────────────────────────────────────────
# Delegates to sub-projects. Does not conflict with report/Makefile.
#
# Web application targets:
#   make web-install      Install dependencies for all web services
#   make web-dev          Start all three web services in parallel (dev mode)
#   make web-typecheck    Type-check backend and frontend
#   make web-build        Production build for backend and frontend
#
# Report targets (delegate to report/Makefile):
#   make report           Build the PDF report
#   make report-docx      Build the DOCX report
#   make report-clean     Clean report build artefacts
#   make report-deps      Install report dependencies
#
# Sandbox/Experiments targets:
#   make sandbox-init              Initialize sandbox student environment venv
#   make sandbox-init-whisper      Install Whisper Python deps
#   make sandbox-whispercpp-init   Clone/build/download whisper.cpp model
#   make sandbox-activate          Activate sandbox student environment
#   make sandbox-test-face         Run face detection test in sandbox
#   make sandbox-test-speech       Run speech recognition test in sandbox
#   make experiments-init          Initialize experiments venv
#   make experiments-dev           Enter experiments development environment

VENV       := web_application/model_service/.venv
PYTHON     := $(VENV)/bin/python
MODEL_DIR  := web_application/model_service

SANDBOX_DIR := sandbox/student_taurajgreig
SANDBOX_VENV := $(SANDBOX_DIR)/venv
SANDBOX_PYTHON := $(SANDBOX_VENV)/bin/python

# whisper.cpp paths inside sandbox
WHISPERCPP_DIR := $(SANDBOX_DIR)/vendor/whisper.cpp
WHISPERCPP_BIN := $(WHISPERCPP_DIR)/build/bin/whisper-cli
WHISPERCPP_MODEL := $(WHISPERCPP_DIR)/models/ggml-base.en.bin

EXP_VENV := experiments/venv
EXP_PYTHON := $(EXP_VENV)/bin/python

.PHONY: web-install web-dev web-typecheck web-build \
        report report-docx report-clean report-deps \
        sandbox-init sandbox-init-whisper sandbox-whispercpp-clone sandbox-whispercpp-build sandbox-whispercpp-model sandbox-whispercpp-init sandbox-activate sandbox-test-face sandbox-test-speech \
        experiments-init experiments-dev

# ── Web application ───────────────────────────────────────────────────────────

$(VENV):
	python3 -m venv $(VENV)

web-install: $(VENV)
	$(PYTHON) -m pip install -r $(MODEL_DIR)/requirements.txt
	cd web_application/backend && npm install
	cd web_application/frontend && npm install
	cd web_application/frontend && npx weave sync

web-dev:
	@echo "Starting model_service, backend, and frontend..."
	cd $(MODEL_DIR) && $(abspath $(PYTHON)) main.py & \
	cd web_application/backend && npm run dev & \
	cd web_application/frontend && npm run dev & \
	wait

web-typecheck:
	cd web_application/backend && npm run typecheck
	cd web_application/frontend && npx weave sync
	cd web_application/frontend && npx svelte-kit sync && npx tsc --noEmit

web-build:
	cd web_application/backend && npm run build
	cd web_application/frontend && npx weave sync
	cd web_application/frontend && npm run build

# ── Report ────────────────────────────────────────────────────────────────────

report:
	$(MAKE) -C report

report-docx:
	$(MAKE) -C report docx

report-clean:
	$(MAKE) -C report clean

report-deps:
	$(MAKE) -C report deps

# ── Sandbox/Student Environment ──────────────────────────────────────────────

$(SANDBOX_VENV):
	python3 -m venv $(SANDBOX_VENV)

sandbox-init: $(SANDBOX_VENV)
	$(SANDBOX_PYTHON) -m pip install --upgrade pip
	$(SANDBOX_PYTHON) -m pip install -r $(SANDBOX_DIR)/requirements.txt
	@echo "✓ Sandbox environment initialized"

sandbox-init-whisper: sandbox-init
	$(SANDBOX_PYTHON) -m pip install -r $(SANDBOX_DIR)/services/requirements-whisper.txt
	@echo "✓ Whisper service dependencies installed"

sandbox-whispercpp-clone:
	@if [ ! -d "$(WHISPERCPP_DIR)" ]; then \
		echo "Cloning whisper.cpp into $(WHISPERCPP_DIR)..."; \
		mkdir -p "$(dir $(WHISPERCPP_DIR))"; \
		git clone https://github.com/ggml-org/whisper.cpp.git "$(WHISPERCPP_DIR)"; \
	else \
		echo "✓ whisper.cpp repo already exists at $(WHISPERCPP_DIR)"; \
	fi

sandbox-whispercpp-build: sandbox-whispercpp-clone
	@if ! command -v cmake >/dev/null 2>&1; then \
		echo "✗ cmake is required. Install it with: brew install cmake"; \
		exit 1; \
	fi
	cd $(WHISPERCPP_DIR) && make
	@echo "✓ whisper.cpp built"

sandbox-whispercpp-model: sandbox-whispercpp-build
	@if [ ! -f "$(WHISPERCPP_MODEL)" ]; then \
		echo "Downloading whisper.cpp model base.en..."; \
		cd $(WHISPERCPP_DIR) && ./models/download-ggml-model.sh base.en; \
	else \
		echo "✓ whisper.cpp model already exists: $(WHISPERCPP_MODEL)"; \
	fi

sandbox-whispercpp-init: sandbox-whispercpp-model
	@echo "✓ whisper.cpp ready"
	@echo "  Binary: $(WHISPERCPP_BIN)"
	@echo "  Model:  $(WHISPERCPP_MODEL)"

sandbox-activate:
	bash --init-file sandbox-activate.sh -i

sandbox-test-face: sandbox-init
	$(SANDBOX_PYTHON) $(SANDBOX_DIR)/face_detection_test.py

sandbox-test-speech: sandbox-init sandbox-whispercpp-init
	$(SANDBOX_PYTHON) $(SANDBOX_DIR)/speech_recognition_test.py

# ── Experiments Environment ──────────────────────────────────────────────────

$(EXP_VENV):
	python3 -m venv $(EXP_VENV)

experiments-init: $(EXP_VENV)
	@echo "Delegating to experiments/Makefile..."
	$(MAKE) -C experiments init

experiments-dev:
	bash --init-file experiments/activate.sh -i