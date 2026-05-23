SHELL := /bin/bash

.PHONY: dev dev-services dev-harness dev-backend dev-frontend install install-training open kill crop-faces test-face-cropper \
        train train-list train-clean

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

# ──────────────────────────────────────────────────────────────────────────
# Training pipeline (v2). See TRAINING.md for the full layout.
#
# `make install-training`  pip install pipeline deps + git-weave sync
#                          vendored repos + symlink POSTER_V2 into the
#                          posterplus model dir so its imports resolve
# `make train`             run every (dataset, model, config) triple in
#                          pipeline/train.py's RUNS list
# `make train-list`        print the runs that would execute, no training
# `make train-clean`       wipe output/ — cached datasets + run dirs + checkpoints
#
# To skip a run, comment out its line in pipeline/train.py RUNS.
# ──────────────────────────────────────────────────────────────────────────

# Where git-weave clones POSTER_V2 (matches vendor/POSTER_V2.thread).
# The model module imports from pipeline/models/posterplus/POSTER_V2/, which
# is a symlink we stage below — keeping the cloned repo in vendor/ keeps
# third-party code separated from our pipeline code.
VENDOR_POSTER_V2 := vendor/POSTER_V2
POSTER_V2_LINK   := pipeline/models/posterplus/POSTER_V2

install-training:
	pip install -r pipeline/requirements.txt
	@echo "→ syncing git-weave vendored repos (POSTER_V2, whisper.cpp, …)"
	npx --yes weave sync
	@echo "→ staging $(VENDOR_POSTER_V2) → $(POSTER_V2_LINK)"
	@if [ ! -d "$(VENDOR_POSTER_V2)" ]; then \
		echo "✗ $(VENDOR_POSTER_V2) missing — weave sync did not clone it. Check vendor/POSTER_V2.thread."; \
		exit 1; \
	fi
	rm -rf "$(POSTER_V2_LINK)"
	ln -s "../../../$(VENDOR_POSTER_V2)" "$(POSTER_V2_LINK)"
	@echo "✓ training pipeline ready. Run \`make train-list\` to see the declared runs."

train:
	python -m pipeline.train

train-list:
	@python -c "from pipeline.train import RUNS; \
print(f'{len(RUNS)} run(s) declared in pipeline/train.py:'); \
[print(f'  {d.NAME:24} x {m.__name__.rsplit(\".\",1)[-1]:14} x {c.NAME}') for (d,m,c) in RUNS]"

train-clean:
	rm -rf output/
	@echo "wiped output/ (cached datasets + run dirs + checkpoints)"
