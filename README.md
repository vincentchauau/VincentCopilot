# 💫 Vincent Copilot

AI-powered code generation and completion tool using HuggingFace models.

## ✨ Features

- **Real-time Inline Completions** - Automatic as-you-type code suggestions
- **25+ Languages** - Python, JavaScript, TypeScript, Java, C++, Go, Rust, and more
- **Code Analysis** - Security scanning, complexity analysis, design patterns
- **Advanced Testing** - Auto-generate tests with edge cases and mocks
- **REST API** - 30+ endpoints for custom integrations
- **100% Free & Open Source** - No subscription, complete privacy
- **Self-Hosted** - Run locally, offline, with full data control
- **Customizable** - Fine-tune models, use your own data, full API access

## Project Structure

```
VincentCopilot/
├── backend/
│   ├── api_server.py              # Flask API server
│   ├── finetune.py                # Model fine-tuning
│   ├── train.py                   # Training script
│   ├── generate.py                # Code generation
│   ├── vincent-cli.py             # CLI tool
│   ├── config.yaml                # Training config
│   ├── language_framework_support.py  # Multi-language support
│   └── static/chat.html           # Web UI
├── frontend/                      # VS Code extension (TypeScript)
└── requirements.txt               # Python dependencies
```

## Setup & Usage

### 1. Install Dependencies
```bash
# Create virtual environment with uv
uv venv
source .venv/Scripts/activate  # Windows Git Bash
# source .venv/bin/activate    # Linux/Mac

# Install packages
uv pip install -r requirements.txt
```

### 2. Start API Server
```bash
cd backend
python api_server.py
```
✨ Access web interface at: http://localhost:5000/static/chat.html

### 3. Use CLI Tool
```bash
python backend/vincent-cli.py --prompt "def hello():"
```

### 4. Fine-tune Model (Optional)
```bash
python backend/finetune.py --config config.yaml
```

### 5. VS Code Extension (Optional)

**Build the extension:**
```bash
cd frontend
npm install
npm run compile
```

**Run in development mode:**
1. Open the `frontend` folder in VS Code
2. Press `F5` to launch Extension Development Host
3. In the new window, open a code file (Python, JS, TS, Java, etc.)
4. **Start typing** - Inline suggestions appear automatically as you type!
5. **Press Tab** to accept suggestions, **Esc** to dismiss
6. **Manual trigger:** `Ctrl+Shift+Space` or command palette → `Vincent Copilot: Trigger Completion`

**Features:**
- ✅ **Real-time inline suggestions** - Automatic as-you-type completions (NEW!)
- ✅ **25+ language support** - Python, JS, TS, Java, C++, Go, Rust, and more
- ✅ **Status bar indicator** - Shows API health (⚡Vincent, ⚠️Error, ❌Offline)
- ✅ **Context detection** - Understands comments, docstrings, and code structure
- ✅ **Configurable** - Adjust delay, tokens, API URL in VS Code settings

**Note:** Make sure the API server is running at http://localhost:5000 before using the extension.