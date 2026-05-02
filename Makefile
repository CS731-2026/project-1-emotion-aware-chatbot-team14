.PHONY: dev dev-harness dev-backend dev-frontend install open

dev:
	make -j3 --keep-going dev-harness dev-backend dev-frontend

dev-harness:
	cd application/model_service && uvicorn app:app --host 0.0.0.0 --port 8000 --reload

dev-backend:
	cd application/backend && npm run dev

dev-frontend:
	cd application/frontend && npm run dev

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
	lsof -ti tcp:5173 | xargs kill -9 2>/dev/null || true
	lsof -ti tcp:8000 | xargs kill -9 2>/dev/null || true
	@echo "Ports 3000 / 5173 / 8000 cleared"
