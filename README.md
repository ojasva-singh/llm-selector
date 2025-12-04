# LLM Selector Application

Intelligent query routing system that analyzes requests and sends them to optimal LLM providers with an automated fallback mechanism.

## Features

- **Smart Query Analysis**: Automatically detects reasoning, fast, or cost-optimized queries
- **Multi-Provider Support**: Anthropic, OpenAI, Google, Groq
- **Fallback Mechanism**: Automatic retry with exponential backoff and provider fallback
- **Cost Optimization**: Routes long responses to cost-effective models
- **Configuration-Driven**: Easy customization via YAML config

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ojasva-singh/llm-selector.git
cd llm-selector
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
# Use any text editor (nano, vim, VS Code, etc.)
nano .env
```

Add your API keys to `.env`:
```bash
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here
GROQ_API_KEY=your_groq_key_here
```

### 5. Run the Application

```bash
python main.py
# or
python3 main.py
```

## Configuration

Edit `config.yaml` to customize:
- Primary and fallback models per query type
- Routing keywords and thresholds
- Retry behavior
- Cost parameters

## Architecture

```
Query → Analyzer → Router → Primary Model
                           ↓ (on failure)
                      Fallback Model 1
                           ↓ (on failure)
                      Fallback Model 2
```