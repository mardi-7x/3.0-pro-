# 🤖 Groq-Powered GPT-Style AI Telegram Chatbot

> A production-grade, ultra-fast, context-aware Telegram AI assistant powered by the **Groq LPU™ Inference Engine**, featuring streaming message delivery, bilingual (English & Bengali) automatic adaptation, SQLite conversational memory, deterministic mathematical validation, and Render Web Service deployment readiness.

---

## 📑 Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Key Features](#2-key-features)
3. [Project Directory Structure](#3-project-directory-structure)
4. [Environment Variables](#4-environment-variables)
5. [Telegram BotFather Setup](#5-telegram-botfather-setup)
6. [Render Deployment Guide (Step-by-Step)](#6-render-deployment-guide-step-by-step)
7. [Local Development & Testing](#7-local-development--testing)
8. [Hallucination Reduction & Verification Safeguards](#8-hallucination-reduction--verification-safeguards)
9. [Polling vs. Webhook Architecture](#9-polling-vs-webhook-architecture)
10. [Limitations & Known Failure Modes](#10-limitations--known-failure-modes)
11. [Production Deployment Checklist](#11-production-deployment-checklist)

---

## 1. Architecture Overview

```
                                ┌─────────────────────────┐
                                │   Telegram Cloud API    │
                                └───────────┬─────────────┘
                                            │ Long Polling
                                            ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         Render Web Service                               │
│                                                                          │
│   ┌───────────────────────────┐         ┌────────────────────────────┐  │
│   │   Flask Health Server     │         │   TelegramAIAssistant      │  │
│   │   (Background Thread)     │         │   (Main Polling Thread)    │  │
│   │   GET / & GET /health     │         │   • Commands (/start, etc) │  │
│   │   Binds to 0.0.0.0:$PORT  │         │   • Sliding Rate Limiter   │  │
│   └───────────────────────────┘         └─────────────┬──────────────┘  │
│                                                       │                 │
│                 ┌─────────────────────────────────────┴─────────────┐   │
│                 ▼                                                   ▼   │
│   ┌───────────────────────────┐                       ┌─────────────┴──┐│
│   │ Deterministic Math Guard  │                       │ DatabaseManager││
│   │ Safe AST Arithmetic Eval  │                       │ SQLite Memory  ││
│   └─────────────┬─────────────┘                       │ User Isolation ││
│                 │                                     └────────────────┘│
│                 ▼                                                       │
│   ┌───────────────────────────┐                                         │
│   │      GroqChatEngine       │                                         │
│   │ Groq SDK Streaming Client │                                         │
│   │ Backoff + Jitter Retries  │                                         │
│   └─────────────┬─────────────┘                                         │
└─────────────────┼───────────────────────────────────────────────────────┘
                  │ HTTPS Streaming Tokens
                  ▼
   ┌─────────────────────────────┐
   │    Groq LPU™ Cloud API      │
   │  llama-3.3-70b-versatile    │
   └─────────────────────────────┘
```

The system operates using a dual-concurrency design:
1. **Health Check Thread (Flask)**: Binds to `$PORT` (injected dynamically by Render) and serves `/health` and `/` endpoints to satisfy Render's zero-downtime health probes.
2. **Main Assistant Thread (TeleBot)**: Runs Telegram's resilient `infinity_polling` loop with exponential backoff on network dropouts.
3. **Groq Inference Engine**: Streams response deltas and edits Telegram messages on buffered intervals (~0.9s) to simulate ChatGPT-like progressive typing without triggering Telegram 429 Too Many Requests errors.
4. **SQLite Memory Store**: Persists chat history across container restarts with per-user isolation.
5. **Deterministic Math Guard**: Intercepts calculations and computes them using an Abstract Syntax Tree (AST) evaluator, providing guaranteed ground-truth calculations.

---

## 2. Key Features

- **⚡ Blazing Fast LPU Inference**: Powered by Groq's high-throughput `llama-3.3-70b-versatile` or `llama-3.1-8b-instant`.
- **💬 Streaming Message Edits**: Progressively updates Telegram messages with cursor indicator (`▌`), respecting Telegram's strict API rate limits.
- **🇧🇩 Seamless Bilingual Support**: Natively understands and communicates in English, Bengali (বাংলা), and Banglish.
- **🛡️ Strict Hallucination Guardrails**: Implements 10 core hallucination-reduction directives and explicit uncertainty statements when evidence is insufficient.
- **📐 Deterministic Math Evaluation**: Safely parses pure mathematical expressions using Python AST to prevent arithmetic hallucinations.
- **🧠 Isolated Multi-User Memory**: SQLite persistent storage ensures User A's context never bleeds into User B's context.
- **⏱️ Per-User Sliding Rate Limiting**: Protects against message spam and API quota exhaustion.
- **✂️ Smart Markdown Message Chunker**: Splits answers exceeding Telegram's 4096-character limit at clean line/paragraph boundaries, preserving code fences.
- **🏥 Built-in Render Health Check**: Zero-configuration Render Web Service compatibility.

---

## 3. Project Directory Structure

```
├── .env.example              # Placeholder environment configuration template
├── Procfile                  # Render startup declaration (`web: python main.py`)
├── README.md                 # Complete documentation and setup manual
├── requirements.txt          # Production dependencies
├── main.py                   # Core monolithic production application
├── tests/
│   └── test_bot.py           # 15 automated unit and integration tests
└── conversations.db          # SQLite persistent database (created automatically at runtime)
```

---

## 4. Environment Variables

| Variable Name | Required | Default | Description |
| :--- | :---: | :---: | :--- |
| `TELEGRAM_BOT_TOKEN` | **Yes** | — | Bot authentication token from Telegram [@BotFather](https://t.me/BotFather). |
| `GROQ_API_KEY` | **Yes** | — | API key from [Groq Console](https://console.groq.com/keys). |
| `GROQ_MODEL` | No | `llama-3.3-70b-versatile` | Groq model identifier (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`). |
| `MAX_HISTORY_MESSAGES`| No | `20` | Maximum message turns preserved per user in context. |
| `MAX_OUTPUT_TOKENS` | No | `2048` | Maximum token budget for Groq generation. |
| `SYSTEM_PROMPT` | No | Built-in | Custom instruction override. |
| `PORT` | No | `10000` | HTTP port for Render health check endpoint. |
| `DATABASE_PATH` | No | `conversations.db` | File path for SQLite storage. |
| `MAX_REQUESTS_PER_MINUTE` | No | `10` | Maximum requests allowed per user per minute. |
| `STREAM_EDIT_INTERVAL_SECONDS`| No | `0.9` | Telegram edit interval to prevent 429 rate limits. |

---

## 5. Telegram BotFather Setup

1. Open Telegram and search for [@BotFather](https://t.me/BotFather).
2. Send `/newbot`.
3. Choose a display name (e.g. `Groq AI Assistant`).
4. Choose a username ending in `bot` (e.g. `MyGroqAIBot`).
5. BotFather will provide an HTTP API token:
   ```
   123456789:ABCdefGHIjklMNOpqrsTUVwxyz
   ```
   Save this as your `TELEGRAM_BOT_TOKEN`.
6. Configure bot commands in BotFather by sending `/setcommands` and pasting:
   ```
   start - Welcome and overview
   help - Detailed feature guide
   new - Start a fresh conversation
   clear - Erase conversation history
   history - View active conversation depth
   model - View active AI model specifications
   status - System uptime and performance stats
   about - Architecture and safety principles
   ```

---

## 6. Render Deployment Guide (Step-by-Step)

### Option A: Deploy via GitHub (Recommended)

1. Push this repository to your GitHub account:
   ```bash
   git init
   git add .
   git commit -m "feat: Groq Telegram AI Assistant production ready"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/groq-telegram-bot.git
   git push -u origin main
   ```

2. Log in to [Render Dashboard](https://dashboard.render.com/).
3. Click **New +** > **Web Service**.
4. Connect your GitHub repository.
5. Configure the service settings:
   - **Name**: `groq-telegram-ai-bot`
   - **Region**: Choose the region closest to you or Groq's servers (e.g., Oregon, US West or Frankfurt, EU).
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python main.py`
   - **Plan**: `Free` (or Starter for uninterrupted persistence)
6. Under **Advanced** > **Environment Variables**, add:
   - `TELEGRAM_BOT_TOKEN`: Your BotFather token.
   - `GROQ_API_KEY`: Your Groq API key (`gsk_...`).
   - `GROQ_MODEL`: `llama-3.3-70b-versatile` (or `llama-3.1-8b-instant`).
7. Under **Health Check Path**, enter: `/health`.
8. Click **Create Web Service**.
9. Render will clone the repo, install dependencies, run the health check server on port 10000, and launch the Telegram bot.

---

## 7. Local Development & Testing

### 1. Clone & Set Up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and enter your actual TELEGRAM_BOT_TOKEN and GROQ_API_KEY
```

### 3. Run Automated Tests

Execute the 15-test automated test suite:

```bash
python3 -m unittest discover tests
```

### 4. Run the Bot Locally

```bash
python main.py
```

Test the health check in your browser or curl:
```bash
curl http://localhost:10000/health
# {"service":"telegram-ai-bot","status":"healthy"}
```

---

## 8. Hallucination Reduction & Verification Safeguards

1. **No Falsified Information**: The system prompt strictly prohibits inventing citations, URLs, dates, or biographical claims.
2. **Explicit Uncertainty**: When evidence is ambiguous or incomplete, the bot responds: *"I cannot verify this information with the available evidence."* (in Bengali: *"বিদ্যমান প্রমাণের ভিত্তিতে আমি এই তথ্যটি নিশ্চিতভাবে যাচাই করতে পারছি না।"*).
3. **Deterministic AST Math**: Arithmetic queries (e.g. `245 * 192`, `sqrt(144) + 42`) are parsed via Python's Abstract Syntax Tree rather than relying on LLM token probabilities.
4. **No Code Execution Simulation**: The assistant never claims to have executed code or accessed live internet data unless verifiable external integrations were executed.
5. **Prompt Injection Hardening**: System instructions explicitly reject instructions embedded in user messages that attempt to extract system prompts or tamper with boundaries.

---

## 9. Polling vs. Webhook Architecture

| Feature | Long Polling + Health Check (Implemented) | Telegram Webhook |
| :--- | :--- | :--- |
| **Setup Complexity** | Zero-configuration (works immediately anywhere). | Requires public HTTPS domain and SSL certificates. |
| **Render Compatibility** | 100% compatible via background Flask health server. | Requires exposing route handler on Flask. |
| **Local Development** | Runs behind NAT/firewall without ngrok. | Requires reverse proxy or tunnel. |
| **Worker Safety** | Must run 1 process (`python main.py`) to prevent duplicate polling. | Multiple web workers can receive updates. |

*Why this project uses Polling + Health Check*: It combines the simplicity of polling with the strict uptime requirements of Render Web Services without requiring domain verification or ngrok tunnels.

---

## 10. Limitations & Known Failure Modes

1. **Render Free Tier Sleep Policy**: Render's Free Web Service plan spins down after 15 minutes of inactivity if no incoming HTTP requests arrive. To keep it awake, set up a free 10-minute HTTP ping (e.g., using Cron-job.org or UptimeRobot) targeting `https://your-service.onrender.com/health`.
2. **Telegram Edit Frequency Limit**: Telegram allows ~1 edit per second per chat. Editing too frequently triggers HTTP 429. This bot buffers edits to 0.9s intervals.
3. **SQLite Ephemeral Storage on Free Render**: Render Free containers have ephemeral disks; redeployments reset the SQLite file unless a persistent disk is attached (available on Render Paid plans).
4. **Context Window Ceiling**: Very long multi-turn sessions are trimmed to `MAX_HISTORY_MESSAGES` to prevent token budget exhaustion.

---

## 11. Production Deployment Checklist

- [x] Environment variables configured (`TELEGRAM_BOT_TOKEN` & `GROQ_API_KEY`).
- [x] Sensitive tokens verified masked in logs.
- [x] Automated test suite passing (`python3 -m unittest discover tests`).
- [x] Health check path configured as `/health` in Render.
- [x] Single-worker startup command set in Procfile (`web: python main.py`).
- [x] Bot commands registered with Telegram @BotFather.
- [x] Rate limiting verified for multi-user protection.
