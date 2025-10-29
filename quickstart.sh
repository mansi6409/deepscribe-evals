#!/bin/bash

# QuickStart Script for DeepScribe Evals Suite
echo "=========================================="
echo "DeepScribe Evals Suite - Quick Start"
echo "=========================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
python -m pip install --upgrade pip -q
echo "✓ Pip upgraded"

# Install dependencies
echo ""
echo "Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "⚠️  Warning: Some dependencies failed to install"
    echo "Trying with relaxed constraints..."
    pip install python-dotenv pydantic pyyaml spacy scispacy sentence-transformers transformers google-generativeai datasets pandas streamlit plotly tqdm tenacity
fi
echo "✓ Dependencies installed"

# Download scispaCy model
echo ""
echo "Downloading medical NER model..."
pip install https://s3-us-west-2.amazonaws.com/ai2-s3-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz
if [ $? -ne 0 ]; then
    echo "⚠️  Warning: Medical NER model download failed (will use fallback)"
else
    echo "✓ Medical NER model installed"
fi

# Verify .env exists
if [ ! -f ".env" ]; then
    echo ""
    echo "⚠️  Warning: .env file not found"
    echo "Creating .env with default API key..."
    echo "GEMINI_API_KEY=AIzaSyBL79H5tDvRLn-Uo8jExo0_PFjlWw2OvOE" > .env
    echo "✓ .env created"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run evaluation:"
echo "   python run_eval.py --mode fast --num-cases 10"
echo ""
echo "2. Launch dashboard:"
echo "   streamlit run dashboard/app.py"
echo ""
echo "3. Read documentation:"
echo "   cat README.md"
echo ""
echo "=========================================="

