# KV Cache Visualization Makefile

.PHONY: serve run stop clean help

# Default port
PORT ?= 8000

# Python executable
PYTHON := python3

## serve: Start the web server
serve: stop
	@echo "Starting server..."
	@sh -c '$(PYTHON) -m http.server $(PORT) > server.log 2>&1 & echo $$! > server.pid'
	@sleep 1
	@echo ""
	@echo "✨ KV Cache Visualization running!"
	@echo "📍 URL: http://localhost:$(PORT)"
	@echo ""
	@echo "Use 'make stop' to stop the server"

## run: Alias for serve
run: serve

## stop: Stop the web server
stop:
	@sh -c 'test -f server.pid && kill $$(cat server.pid) 2>/dev/null || true'
	@rm -f server.pid

## clean: Clean up server files
clean: stop
	@rm -f server.log server.pid
	@echo "Cleaned up server files"

## help: Show this help message
help:
	@echo "KV Cache Visualization - Memory Growth Demo"
	@echo "==========================================="
	@echo ""
	@echo "Usage:"
	@echo "  make              Start server at http://localhost:8000"
	@echo "  make serve        Start server"
	@echo "  make run          Same as serve"
	@echo "  make stop         Stop the running server"
	@echo "  make clean        Stop server and clean up files"
	@echo "  make help         Show this help message"
	@echo ""
	@echo "Options:"
	@echo "  PORT=8080 make serve    Use custom port (default: 8000)"
	@echo ""
	@echo "Features:"
	@echo "  • Watch KV cache memory explode from 0 to 100M tokens"
	@echo "  • Compare 5 models from 1B to 671B parameters"
	@echo "  • Switch data types (FP32/FP16/BF16/INT8/INT4)"
	@echo "  • Speed controls from 0.5x to 100x"
	@echo ""
	@echo "The visualization demonstrates why LMCache optimizations are critical"
	@echo "as context lengths grow to millions of tokens."

# Default target
.DEFAULT_GOAL := serve
