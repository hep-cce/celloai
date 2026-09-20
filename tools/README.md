# CelloAI Tools - Installation & MCP Configuration Guide

This guide covers installing the required tools and configuring MCP servers for **Claude Code**, **Codex**, and **OpenCode**.

---

## 1. Prerequisites

### Install `uv` (Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### Set Brain API Key

```bash
export BRAIN_API_KEY=YOUR_KEY
```

---

## 2. LLM and MCP Configuration Reference

| Agent | Global LLM Config File | Project Level MCP Config Location |
|-------|-------------|---------------------|
| **Claude Code** | `~/.claude/settings.json` | `.mcp.json` (project-level) |
| **Codex** | `~/.codex/config.toml` | `.codex/config.toml` |
| **OpenCode** | `~/.config/opencode/opencode.json` | `opencode.json` |


---

## 3. Configure Coding Agents to Use Local LLMs

### 3.1 Claude Code

**Install:**
```bash
curl -fsSL https://claude.ai/install.sh | bash
```

**Configure** in `~/.claude/settings.json`:

```json
{
  "apiKeyHelper": "printenv BRAIN_API_KEY",
  "env": {
    "ANTHROPIC_BASE_URL": "https://inference0-api.sdcc.bnl.gov",
    "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "DISABLE_TELEMETRY": "1",
    "DISABLE_ERROR_REPORTING": "1",
    "DISABLE_FEEDBACK_COMMAND": "1",
    "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL": "1"
  },
  "model": "nemotron-3-super-120b",
  "theme": "dark",
  "modelPicker": {
    "replaceBuiltInOptions": true,
    "options": [
      {
        "model": "nemotron-3-super-120b",
        "label": "Nemotron 120B",
        "description": "Primary coding and reasoning model"
      },
      {
        "model": "gpt-oss-120b",
        "label": "GPT-OSS 120B",
        "description": "Alternative large reasoning model"
      },
      {
        "model": "nemotron-3-ultra-550b-nvfp4",
        "label": "Nemotron Ultra 550B",
        "description": "Largest thinking model"
      },
      {
        "model": "gemma-4-26b",
        "label": "Gemma 26B",
        "description": "Vision-capable model exposed by the gateway"
      }
    ]
  }
}
```

---

### 3.2 Codex

**Install:**
```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
```

**Configure** in `~/.codex/config.toml`:

```toml
model_provider = "bnl_api"
model          = "nemotron-3-super-120b"

[model_providers.bnl_api]
name                 = "BNL Inference"
base_url             = "https://inference0-api.sdcc.bnl.gov/v1"
env_key              = "BRAIN_API_KEY"
wire_api             = "responses"
requires_openai_auth = false

[tui.model_availability_nux]
"gpt-5.5" = 4
```

---

### 3.3 OpenCode

**Install:**
```bash
curl -fsSL https://opencode.ai/install | bash
```

**Configure** in `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "bnl": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "BNL Inference",
      "options": {
        "baseURL": "https://inference0-api.sdcc.bnl.gov/v1",
        "apiKey": "{env:BRAIN_API_KEY}"
      },
      "models": {
        "gpt-oss-120b": {
          "name": "gpt-oss-120b"
        },
        "gemma-4-26b": {
          "name": "gemma-4-26b"
        },
        "nemotron-3-super-120b": {
          "name": "nemotron-3-super-120b"
        },
        "nemotron-3-ultra-550b-nvfp4": {
          "name": "nemotron-3-ultra-550b-nvfp4"
        }
      }
    }
  }
}
```

---

## 4. Available MCP Servers

| Server | Description | Entry Point |
|--------|-------------|-------------|
| `brain_vision` | Vision LLM for image analysis | `server_vision_llm.py` |
| `celloai_callgraph` | Call graph analysis | `server_callgraph.py` |
| `celloai_retriever` | Cello code retrieval | `server_cello_retriever.py` |

Check if MCP servers are properly configured by

```bash
claude mcp list
codex mcp list
opencode mcp list
```
