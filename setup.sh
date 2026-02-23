#!/bin/bash

# ============================================================================
# Evidence Engine Setup Script
# ============================================================================

echo "--- Initializing Evidence Engine Setup ---"

# 1. Create Virtual Environment
if [ ! -d "venv" ]; then
    echo "Status: Creating virtual environment..."
    python3 -m venv venv
else
    echo "Status: Virtual environment already exists."
fi

# 2. Activate Environment
source venv/bin/activate

# 3. Upgrade Core Pip Tools
echo "Status: Upgrading pip and installer tools..."
pip install --upgrade pip setuptools wheel

# 4. Install Dependencies with Conflict Resolution
# We force-install these specific versions first to prevent common LangChain/Streamlit conflicts
echo "Status: Installing dependencies (this may take a minute)..."
pip install packaging==24.1 requests==2.32.3

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "❌ Error: requirements.txt not found!"
    exit 1
fi

# 5. Check for Graphviz (System Dependency)
# Required for rendering PRISMA Flow Diagrams
if ! command -v dot &> /dev/null; then
    echo "⚠️  Warning: Graphviz (system binary) not found."
    echo "   PRISMA diagrams will not render without it."
    echo "   Install via: 'brew install graphviz' (Mac) or 'sudo apt install graphviz' (Linux)"
fi

# 6. Initialize Local AI (Ollama)
if command -v ollama &> /dev/null; then
    echo "Status: Ollama detected. Pulling default model (llama3)..."
    ollama pull llama3
else
    echo "⚠️  Note: Ollama not found. Local AI features will be disabled until Ollama is installed."
    echo "   Download it from: https://ollama.ai"
fi

# 7. Final Verification
echo "Status: Running dependency integrity check..."
pip check

echo "-----------------------------------------------"
echo " Setup Complete!"
echo "-----------------------------------------------"
echo "To launch the Evidence Engine:"
echo "1. source venv/bin/activate"
echo "2. streamlit run app.py"
echo "-----------------------------------------------"