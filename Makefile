SHELL := /bin/bash

.PHONY: dev dev-services dev-harness dev-backend dev-frontend install install-training open kill crop-faces test-face-cropper \
        train train-list train-clean deploy-model publish-model publish-models fetch-models compare new-model

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
	@if [ -e "$(POSTER_V2_LINK)" ] || [ -L "$(POSTER_V2_LINK)" ]; then \
		echo "  $(POSTER_V2_LINK) already exists — leaving in place. Remove it manually to re-stage."; \
	else \
		cp -R "$(VENDOR_POSTER_V2)" "$(POSTER_V2_LINK)"; \
		echo "  copied vendor tree into $(POSTER_V2_LINK)"; \
	fi
	@echo "✓ training pipeline ready. Run \`make train-list\` to see the declared runs."

train:
	python -m pipeline.train $(if $(RUN),--run "$(RUN)")

train-list:
	@python -c "from pipeline.runs_loader import load; \
runs=load('runs.yaml'); \
print(f'{len(runs)} enabled run(s) in runs.yaml:'); \
[print(f'  {r.dataset.NAME:24} x {r.model.__name__.rsplit(\".\",1)[-1]:22} x {r.config.NAME}') for r in runs]"

train-clean:
	rm -rf output/
	@echo "wiped output/ (cached datasets + run dirs + checkpoints)"

# Scaffold a new model module under pipeline/models/<id>/.
#   make new-model ID=my_model                 # simple template (30-line)
#   make new-model ID=my_model TEMPLATE=custom # full 5-file layout
new-model:
	@if [ -z "$(ID)" ]; then \
		echo "usage: make new-model ID=<snake_case_id> [TEMPLATE=simple|custom] [FORCE=1]"; \
		exit 2; \
	fi
	@python -m pipeline.cli.new_model "$(ID)" \
		$(if $(TEMPLATE),--template $(TEMPLATE),--template simple) \
		$(if $(FORCE),--force)

# Print a leaderboard of every run under output/run/.
#   make compare                         # everything, sorted by test_acc
#   make compare FILTER=empathbot        # substring filter
#   make compare FILTER=fer2013 TOP=10
#   make compare SORT=val_acc
compare:
	@python -m pipeline.cli.compare \
		$(if $(FILTER),--filter "$(FILTER)") \
		$(if $(SORT),--sort-by $(SORT)) \
		$(if $(TOP),--top $(TOP))

# ──────────────────────────────────────────────────────────────────────────
# Deploy a trained checkpoint into the model_service.
#
# Copies output/run/<RUN>/checkpoints/best.pth → models/<ID>/best.pth and
# adds an entry to application/model_service/models.yaml mapping <ID>
# to the correct service variant. After deploy, set EMOTION_MODEL_ID=<ID>
# in .env (or inline) and `make dev` to use it.
#
# Usage:
#   make deploy-model RUN=fer2013__empathbot_final__thorough__20260524-... ID=empathbot_final
#   make deploy-model RUN=LATEST ID=my_model         # picks newest output/run/ subdir
#   make deploy-model RUN=... ID=... VARIANT=resnet18  # override inferred variant
# ──────────────────────────────────────────────────────────────────────────
deploy-model:
	@if [ -z "$(RUN)" ] || [ -z "$(ID)" ]; then \
		echo "usage: make deploy-model RUN=<run-dir-name> ID=<model-id> [VARIANT=<variant>] [CHECKPOINT=best.pth]"; \
		echo "  RUN=LATEST  picks the most recently created dir under output/run/"; \
		exit 2; \
	fi
	@RUN_RESOLVED="$(RUN)"; \
	if [ "$$RUN_RESOLVED" = "LATEST" ]; then \
		RUN_RESOLVED=$$(ls -t output/run/ | head -1); \
		echo "→ LATEST resolved to: $$RUN_RESOLVED"; \
	fi; \
	python -m pipeline.cli.deploy_model \
		--run "output/run/$$RUN_RESOLVED" \
		--id "$(ID)" \
		$(if $(VARIANT),--variant $(VARIANT)) \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT))

# ──────────────────────────────────────────────────────────────────────────
# Kaggle weights dataset — share trained checkpoints without committing
# binaries. Auth: KAGGLE_USERNAME/KAGGLE_KEY in .env (or kaggle.json).
# Slug from KAGGLE_WEIGHTS_SLUG in .env (default team14/empathbot-checkpoints).
#
#   make publish-models [MESSAGE="…"]   # uploads ALL models/<id>/ as one version
#   make publish-models NEW=1            # first-time dataset create
#   make fetch-models                    # pulls every model into models/
#
# Note: the live model_service auto-fetches from Kaggle when a configured
# model id resolves to a missing local path — usually you don't need to
# run fetch-models manually.
# ──────────────────────────────────────────────────────────────────────────
publish-models:
	python -m pipeline.cli.publish_models \
		$(if $(MESSAGE),--message "$(MESSAGE)") \
		$(if $(NEW),--new-dataset) \
		$(if $(SLUG),--slug $(SLUG))

# Discouraged: publish just one model. Kaggle datasets are atomic, so
# this REPLACES every other model in the dataset with just <id>. Use
# `make publish-models` (plural) unless you have a specific reason.
publish-model:
	@if [ -z "$(ID)" ]; then \
		echo "usage: make publish-model ID=<id>     # DISCOURAGED — use publish-models (plural)"; \
		exit 2; \
	fi
	python -m pipeline.cli.publish_models --id "$(ID)" \
		$(if $(MESSAGE),--message "$(MESSAGE)") \
		$(if $(NEW),--new-dataset) \
		$(if $(SLUG),--slug $(SLUG))

fetch-models:
	python -m pipeline.cli.fetch_models $(if $(SLUG),--slug $(SLUG))
