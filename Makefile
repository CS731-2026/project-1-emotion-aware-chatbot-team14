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

VENV       := web_application/model_service/.venv
PYTHON     := $(VENV)/bin/python
MODEL_DIR  := web_application/model_service

.PHONY: web-install web-dev web-typecheck web-build \
        report report-docx report-clean report-deps

# ── Web application ───────────────────────────────────────────────────────────

$(VENV):
	python3 -m venv $(VENV)

web-install: $(VENV)
	$(PYTHON) -m pip install -r $(MODEL_DIR)/requirements.txt
	cd web_application/backend && npm install
	cd web_application/frontend && npm install

web-dev:
	@echo "Starting model_service, backend, and frontend..."
	cd $(MODEL_DIR) && $(abspath $(PYTHON)) main.py & \
	cd web_application/backend && npm run dev & \
	cd web_application/frontend && npm run dev & \
	wait

web-typecheck:
	cd web_application/backend && npm run typecheck
	cd web_application/frontend && npx svelte-kit sync && npx tsc --noEmit

web-build:
	cd web_application/backend && npm run build
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
