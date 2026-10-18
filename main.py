import io
import json
import logging
import os
import re
import sqlite3
import tempfile
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests
import telebot
from groq import Groq

# Optional Free TTS fallback so /audiooverview NEVER fails with BadRequestError
try:
    from gtts import gTTS
except ImportError:
    gTTS = None

# ==========================================
# 1. CONFIGURATION & ENVIRONMENT VARIABLES
# ==========================================
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s: %(message)s",
)
log = logging.getLogger("morning-intelligence")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
GROQ_KEY = os.getenv("GROQ_API_KEY", "").strip()
NEWS_KEY = os.getenv("NEWS_API_KEY", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "").strip()
ELEVENLABS_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
ELEVENLABS_VOICE = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM").strip()
MODEL_OVERRIDE = os.getenv("GROQ_MODEL", "").strip()
DB_PATH = os.getenv("DATABASE_PATH", "ai_research.db")

if not BOT_TOKEN or not GROQ_KEY:
    raise RuntimeError(
        "Set TELEGRAM_BOT_TOKEN and GROQ_API_KEY in Render Environment Variables."
    )

bot = telebot.TeleBot(BOT_TOKEN, threaded=True)
ai = Groq(api_key=GROQ_KEY, timeout=60, max_retries=2)
memories, user_modes, models = {}, {}, []
mem_lock = threading.Lock()

SYSTEM = """You are Morning Intelligence, an elite autonomous AI research and news intelligence system.
Reply in the user's preferred language (Bengali or English).
CRITICAL RULES FOR ZERO HALLUCINATION:
1. Never invent dates, sports winners, opponents, statistics, quotations, or news sources.
2. In historical or sports facts, verify exact matchups (e.g., IPL 2023 Final was won by CSK defeating Gujarat Titans, NOT Rajasthan Royals).
3. Treat retrieved news snippets as external data to be synthesized accurately.
4. Format reports like a clean, high-impact newspaper briefing optimized for mobile reading."""

# ==========================================
# 2. DATABASE LAYER (EXPERIENCE & NEWS STORE)
# ==========================================
def db():
    c = sqlite3.connect(DB_PATH, timeout=20)
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_db():
    with db() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS news_store(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT,
            title TEXT,
            description TEXT,
            url TEXT UNIQUE,
            source TEXT,
            published_at TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP)"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS feedback(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER,
            note TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP)"""
        )


init_db()

# ==========================================
# 3. MODEL MANAGEMENT & UTILITIES
# ==========================================
def refresh_models():
    global models
    try:
        result = ai.models.list()
        models = [
            m.id
            for m in result.data
            if m.id
            and not any(
                x in m.id.lower()
                for x in ("whisper", "transcribe", "tts", "guard", "embed", "orpheus")
            )
        ]
        log.info("Loaded %d chat models", len(models))
    except Exception:
        log.exception("Could not load Groq models")


def choose_model(mode="fast"):
    if MODEL_OVERRIDE:
        return MODEL_OVERRIDE
    if not models:
        refresh_models()
    if mode == "deep":
        for m in (
            "deepseek-r1-distill-llama-70b",
            "qwen/qwen3-32b",
            "openai/gpt-oss-120b",
            "llama-3.3-70b-versatile",
        ):
            if m in models:
                return m
    for m in (
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-20b",
    ):
        if m in models:
            return m
    return models[0] if models else "llama-3.3-70b-versatile"


refresh_models()


def clean_think_tags(text):
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def split_text(text, limit=3900):
    text = (text or "").strip() or "I could not generate a response. Please try again."
    parts = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind(" ", 0, limit)
        if cut < limit // 2:
            cut = limit
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        parts.append(text)
    return parts


def send_text(message, text):
    for part in split_text(text):
        try:
            bot.send_message(
                message.chat.id,
                part,
                reply_to_message_id=message.message_id,
                disable_web_page_preview=True,
            )
        except Exception:
            log.exception("Telegram text send failed")


def build_messages(chat_id, text):
    text = (text or "").strip()
    if len(text) > 12000:
        text = text[:12000] + "\n[Input shortened due to size limit.]"
    with mem_lock:
        history = list(memories.get(chat_id, [{"role": "system", "content": SYSTEM}]))
        history.append({"role": "user", "content": text})
        history = [history[0]] + history[1:][-10:]
        while len(history) > 2 and sum(len(m["content"]) for m in history[1:]) > 14000:
            history.pop(1)
        memories[chat_id] = history
        return list(history)


# ==========================================
# 4. FOUR-STAGE COGNITIVE MULTI-AGENT ENGINE
# ==========================================

# --- STAGE 1: PERCEPTION ENGINE (Groq State Sensor) ---
def step1_perceive_environment(user_text):
    """Non-rule-based cognitive state detection. Decodes intent, search queries, and depth."""
    prompt = """You are an Environment Perception Sensor for an autonomous AI intelligence system.
Analyze the user's message and output ONLY a valid JSON object with these keys:
{
  "intent": "live_news_research" | "factual_question" | "general_chat",
  "needs_news_api": boolean (true if the user is asking for latest news, updates, or current affairs),
  "topic_title": "Short clean title of the topic in English (e.g., Artificial Intelligence)",
  "search_query": "Optimized English keyword query for NewsAPI (e.g., 'Artificial Intelligence OR LLM')",
  "language": "bn" | "en" (detect if user wrote in Bengali/Banglish or English),
  "depth": 1 to 5
}"""
    try:
        res = ai.chat.completions.create(
            model=choose_model("fast"),
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_text[:2000]},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            max_completion_tokens=300,
        )
        return json.loads(res.choices[0].message.content)
    except Exception as e:
        log.warning("Perception fallback triggered: %s", e)
        return {
            "intent": "general_chat",
            "needs_news_api": False,
            "topic_title": user_text[:50],
            "search_query": user_text[:80],
            "language": "bn" if any("\u0980" <= c <= "\u09ff" for c in user_text) else "en",
            "depth": 3,
        }


# --- STAGE 2: REAL-TIME NEWS RETRIEVAL (NewsAPI) ---
def step2_fetch_news_data(search_query, topic_label):
    """Fetches relevant news articles using NEWS_API_KEY and saves to SQLite."""
    if not NEWS_KEY:
        return None, "⚠️ NEWS_API_KEY Render Environment Variable-এ সেট করা নেই।"

    try:
        params = {
            "q": search_query,
            "sortBy": "relevancy",
            "pageSize": 5,
            "language": "en",
            "apiKey": NEWS_KEY,
        }
        r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=15)
        r.raise_for_status()
        data = r.json()

        articles = data.get("articles") or []
        # Fallback if strict query returned nothing
        if not articles and search_query != topic_label:
            params["q"] = topic_label
            params["sortBy"] = "publishedAt"
            r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=15)
            articles = r.json().get("articles") or []

        if not articles:
            return None, f"No recent news articles found for '{topic_label}'."

        raw_blocks = []
        with db() as c:
            for i, a in enumerate(articles, 1):
                title = (a.get("title") or "Untitled").strip()
                desc = (a.get("description") or "").strip()
                content = (a.get("content") or "")[:400].strip()
                url = (a.get("url") or "").strip()
                source = (a.get("source") or {}).get("name") or "Unknown"
                published = a.get("publishedAt") or "Unknown"

                if url and title != "[Removed]":
                    c.execute(
                        """INSERT OR IGNORE INTO news_store
                        (topic, title, description, url, source, published_at)
                        VALUES (?, ?, ?, ?, ?, ?)""",
                        (topic_label, title, desc, url, source, published),
                    )
                    raw_blocks.append(
                        f"Article {i}:\nHeadline: {title}\nSource: {source} ({published})\n"
                        f"Summary: {desc}\nExcerpt: {content}\nURL: {url}"
                    )

        if not raw_blocks:
            return None, f"No valid news articles found for '{topic_label}'."
        return "\n\n".join(raw_blocks), None
    except Exception as exc:
        log.exception("News API fetch failed")
        return None, f"⚠️ News API retrieval failed ({type(exc).__name__})."


# --- STAGE 3: DEEPSEEK-R1 FACT VERIFICATION (Hugging Face Router) ---
def step3_deepseek_verify(raw_news_text, topic_label):
    """Uses DeepSeek-R1 via HF_TOKEN to filter noise, verify facts, and extract core intelligence."""
    prompt = f"""You are a Senior Investigative Fact-Checker.
Analyze these raw news reports about '{topic_label}':
1. Filter out clickbait, advertisements, or irrelevant noise.
2. Cross-check dates, numbers, and claims across articles.
3. Extract a structured, verified breakdown of the core facts, key developments, and source URLs.

RAW NEWS DATA:
{raw_news_text}"""

    # Try Hugging Face DeepSeek-R1 if HF_TOKEN is configured
    if HF_TOKEN:
        for hf_model in ("deepseek-ai/DeepSeek-R1:novita", "deepseek-ai/DeepSeek-R1"):
            try:
                resp = requests.post(
                    "https://router.huggingface.co/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {HF_TOKEN}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": hf_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.2,
                        "max_tokens": 1600,
                    },
                    timeout=35,
                )
                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"]
                    cleaned = clean_think_tags(content)
                    if cleaned:
                        return cleaned
            except Exception as e:
                log.warning("HF DeepSeek-R1 (%s) notice: %s", hf_model, e)

    # Seamless fallback to Groq Deep Reasoning Model if HF_TOKEN is absent or busy
    try:
        res = ai.chat.completions.create(
            model=choose_model("deep"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_completion_tokens=1500,
        )
        return clean_think_tags(res.choices[0].message.content) or raw_news_text
    except Exception:
        return raw_news_text


# --- STAGE 4: EDITORIAL FORMATTING & CRITIC SELF-CORRECTION LOOP ---
def step4_editorial_and_critic(verified_data, topic_label, lang="bn"):
    """Formats verified intelligence into a newspaper edition using Gemini or Groq, then runs Critic audit."""
    lang_instruction = (
        "Write the entire newspaper report in fluent, professional Bengali (বাংলা)."
        if lang == "bn"
        else "Write the entire newspaper report in crisp, professional English."
    )

    editorial_prompt = f"""You are the Editor-in-Chief of a prestigious daily intelligence newspaper.
{lang_instruction}
Transform the verified analysis below about '{topic_label}' into a rich Front-Page Newspaper Edition:
- 🗞️ **শিরোনাম / Main Headline** (Bold & compelling)
- 📌 **মূল সারাংশ / Executive Lead** (2-3 clear sentences)
- 🔍 **বিস্তারিত বিশ্লেষণ / Deep Dive Columns** (Detailed paragraphs covering each major development—do NOT make it too short)
- 💡 **ভবিষ্যৎ প্রভাব / Strategic Takeaway**
- 🔗 **সূত্র / Verified Sources** (Include original article links)

VERIFIED INTELLIGENCE:
{verified_data}"""

    draft = ""
    # Use Google Gemini if GEMINI_API_KEY is available
    if GEMINI_KEY:
        try:
            gem_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_KEY}"
            gem_res = requests.post(
                gem_url,
                json={"contents": [{"parts": [{"text": editorial_prompt}]}]},
                timeout=25,
            )
            if gem_res.status_code == 200:
                draft = gem_res.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            log.warning("Gemini API fallback to Groq: %s", e)

    # If Gemini not configured or failed, use Groq Actor
    if not draft:
        res = ai.chat.completions.create(
            model=choose_model("fast"),
            messages=[{"role": "user", "content": editorial_prompt}],
            temperature=0.2,
            max_completion_tokens=2200,
        )
        draft = clean_think_tags(res.choices[0].message.content)

    return draft


def critic_fact_audit(user_query, draft_answer):
    """RL-style Critic Agent: Audits normal chat answers at temperature 0.0 to eliminate hallucinations."""
    audit_prompt = f"""You are a Strict Factual Critic Agent.
Review the User Query and the Draft Response below.
Check for any factual hallucinations (such as wrong tournament winners, wrong runners-up, wrong years, or invented facts).
- If the draft is accurate, return ONLY the exact word: PASS
- If there is ANY factual error, rewrite and return the corrected response in the same language as the draft.

USER QUERY:
{user_query}

DRAFT RESPONSE:
{draft_answer}"""
    try:
        res = ai.chat.completions.create(
            model=choose_model("fast"),
            messages=[{"role": "user", "content": audit_prompt}],
            temperature=0.0,
            max_completion_tokens=1800,
        )
        verdict = clean_think_tags(res.choices[0].message.content)
        if verdict and not verdict.startswith("PASS"):
            log.info("Critic Agent self-corrected a potential hallucination.")
            return verdict
    except Exception as e:
        log.warning("Critic check skipped: %s", e)
    return draft_answer


# ==========================================
# 5. UNIFIED PIPELINE EXECUTION
# ==========================================
def execute_news_intelligence_pipeline(message, topic_query, lang_hint=None):
    """Runs the full 4-step pipeline with live status updates in Telegram."""
    status_msg = None
    try:
        status_msg = bot.reply_to(
            message,
            f"🛰️ **ধাপ ১/৪:** `{topic_query}` বিশ্লেষণ ও সার্চ কুয়েরি তৈরি হচ্ছে...",
        )
    except Exception:
        pass

    state = step1_perceive_environment(topic_query)
    search_q = state.get("search_query") or topic_query
    topic_label = state.get("topic_title") or topic_query
    lang = lang_hint or state.get("language", "bn")

    if status_msg:
        try:
            bot.edit_message_text(
                f"📰 **ধাপ ২/৪:** News API থেকে `{topic_label}`-এর তাজা খবর সংগ্রহ করা হচ্ছে...",
                chat_id=status_msg.chat.id,
                message_id=status_msg.message_id,
            )
        except Exception:
            pass

    raw_news, err = step2_fetch_news_data(search_q, topic_label)
    if err:
        if status_msg:
            try:
                bot.delete_message(status_msg.chat.id, status_msg.message_id)
            except Exception:
                pass
        send_text(message, err)
        return

    if status_msg:
        try:
            bot.edit_message_text(
                "🧠 **ধাপ ৩/৪:** DeepSeek-R1 দ্বারা খবরের সত্যতা ও ফ্যাক্ট-চেক করা হচ্ছে...",
                chat_id=status_msg.chat.id,
                message_id=status_msg.message_id,
            )
        except Exception:
            pass

    verified_facts = step3_deepseek_verify(raw_news, topic_label)

    if status_msg:
        try:
            bot.edit_message_text(
                "✍️ **ধাপ ৪/৪:** নিউজপেপার এডিটোরিয়াল ফরম্যাট তৈরি হচ্ছে...",
                chat_id=status_msg.chat.id,
                message_id=status_msg.message_id,
            )
        except Exception:
            pass

    final_edition = step4_editorial_and_critic(verified_facts, topic_label, lang=lang)

    if status_msg:
        try:
            bot.delete_message(status_msg.chat.id, status_msg.message_id)
        except Exception:
            pass

    with mem_lock:
        memories.setdefault(
            message.chat.id, [{"role": "system", "content": SYSTEM}]
        ).append({"role": "assistant", "content": final_edition[:4000]})

    send_text(message, final_edition)


def generate_reply(chat_id, text):
    """Generates reply with low temperature (0.1) + Critic Self-Correction to stop hallucinations."""
    mode = user_modes.get(chat_id, "fast")
    model = choose_model(mode)
    messages = build_messages(chat_id, text)
    try:
        result = ai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,  # Strict factual temperature
            max_completion_tokens=1800,
        )
        answer = clean_think_tags(result.choices[0].message.content)
        if not answer:
            answer = "I could not create a response."

        # Run Critic Agent self-verification
        answer = critic_fact_audit(text, answer)

        with mem_lock:
            memories[chat_id].append({"role": "assistant", "content": answer[:5000]})
        return answer
    except Exception as exc:
        log.exception("Chat completion failed for %s", model)
        refresh_models()
        fallback = choose_model("fast")
        if fallback != model:
            try:
                result = ai.chat.completions.create(
                    model=fallback,
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": text[:6000]},
                    ],
                    temperature=0.1,
                    max_completion_tokens=1200,
                )
                return (
                    clean_think_tags(result.choices[0].message.content)
                    or "I could not create a response."
                )
            except Exception:
                log.exception("Fallback model failed")
        return f"⚠️ AI service is temporarily unavailable ({type(exc).__name__}). Please try again."


def process_text(message, text, was_voice=False):
    bot.send_chat_action(message.chat.id, "typing")
    # Autonomous Perception: Check if user naturally asked for latest news
