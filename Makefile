SHELL := /bin/bash

.PHONY: dev dev-services dev-harness dev-backend dev-frontend install open kill crop-faces test-face-cropper

dev: kill
	$(MAKE) -j3 --keep-going dev-harness dev-backend dev-frontend

dev-harness:
	set -o pipefail; cd application/model_service && PYTHONUNBUFFERED=1 uvicorn app:app --host 0.0.0.0 --port 8000 --reload 2>&1 | awk '{ print "[harness] " $$0; fflush(); }'

dev-backend:
	lsof -ti tcp:3001 | xargs kill -9 2>/dev/null || true
	set -o pipefail; cd application/backend && npm run dev 2>&1 | awk '{ print "[backend] " $$0; fflush(); }'

dev-frontend:
	set -o pipefail; cd application/frontend && npm run dev 2>&1 | awk '{ print "[frontend] " $$0; fflush(); }'

install:
	cd application/backend && npm install
	cd application/frontend && npm install
	cd application/model_service && pip install -r requirements.txt

# macOS: open each service in a new Terminal tab
open:
	osascript -e 'tell app "Terminal" to do script "cd $(CURDIR)/application/model_service && uvicorn app:app --reload"'
	osascript -e 'tell app "Terminal" to do script "cd $(CURDIR)/application/backend && npm run dev"'
	osascript -e 'tell app "Terminal" to do script "cd $(CURDIR)/application/frontend && npm run dev"'

kill:
	lsof -ti tcp:3000 | xargs kill -9 2>/dev/null || true
	lsof -ti tcp:3001 | xargs kill -9 2>/dev/null || true
	lsof -ti tcp:5173 | xargs kill -9 2>/dev/null || true
	lsof -ti tcp:8000 | xargs kill -9 2>/dev/null || true
	@echo "Ports 3000 / 3001 / 5173 / 8000 cleared"

# Face cropper — see face_cropper/README.md for details.
# Wraps the production face detector for batch dataset preprocessing.
crop-faces:
	@if [ -z "$(INPUT)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "usage: make crop-faces INPUT=<dir> OUTPUT=<dir> [RESIZE=224] [PADDING=0.1]"; \
		exit 2; \
	fi
	python face_cropper.py crop-dir "$(INPUT)" "$(OUTPUT)" --recursive \
		$(if $(RESIZE),--resize $(RESIZE)) \
		$(if $(PADDING),--padding $(PADDING)) \
		--skip-existing \
		--report "$(OUTPUT)/_crop_report.json"

test-face-cropper:
	python face_cropper/test_face_cropper.py $(IMAGE)
