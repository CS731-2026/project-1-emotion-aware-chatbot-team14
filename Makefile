SHELL := /bin/bash

# Per-worktree port assignments live in .env.ports (gitignored, written by
# scripts/worktree-add.sh). Falls back to the defaults below for the main
# checkout / fresh clones.
-include .env.ports

BACKEND_PORT  ?= 3001
FRONTEND_PORT ?= 5173
MODEL_PORT    ?= 8000

.PHONY: dev dev-services dev-harness dev-backend dev-frontend install open kill ports

dev: kill
	$(MAKE) -j3 --keep-going dev-harness dev-backend dev-frontend

dev-harness:
	set -o pipefail; cd application/model_service && PYTHONUNBUFFERED=1 uvicorn app:app --host 0.0.0.0 --port $(MODEL_PORT) --reload 2>&1 | awk '{ print "[harness] " $$0; fflush(); }'

dev-backend:
	lsof -ti tcp:$(BACKEND_PORT) | xargs kill -9 2>/dev/null || true
	set -o pipefail; cd application/backend && npm run dev 2>&1 | awk '{ print "[backend] " $$0; fflush(); }'

dev-frontend:
	set -o pipefail; cd application/frontend && npm run dev -- --port $(FRONTEND_PORT) 2>&1 | awk '{ print "[frontend] " $$0; fflush(); }'

install:
	cd application/backend && npm install
	cd application/frontend && npm install
	cd application/model_service && pip install -r requirements.txt

# macOS: open each service in a new Terminal tab
open:
	osascript -e 'tell app "Terminal" to do script "cd $(CURDIR)/application/model_service && uvicorn app:app --port $(MODEL_PORT) --reload"'
	osascript -e 'tell app "Terminal" to do script "cd $(CURDIR)/application/backend && npm run dev"'
	osascript -e 'tell app "Terminal" to do script "cd $(CURDIR)/application/frontend && npm run dev -- --port $(FRONTEND_PORT)"'

kill:
	lsof -ti tcp:$(BACKEND_PORT)  | xargs kill -9 2>/dev/null || true
	lsof -ti tcp:$(FRONTEND_PORT) | xargs kill -9 2>/dev/null || true
	lsof -ti tcp:$(MODEL_PORT)    | xargs kill -9 2>/dev/null || true
	@echo "Cleared ports: backend=$(BACKEND_PORT) frontend=$(FRONTEND_PORT) model=$(MODEL_PORT)"

ports:
	@echo "backend  = $(BACKEND_PORT)"
	@echo "frontend = $(FRONTEND_PORT)"
	@echo "model    = $(MODEL_PORT)"
