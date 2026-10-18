import os
import re
import json
import time
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import telebot
from groq import Groq

# 1. Logging & Environment Variables
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("morning-core")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
GROQ_KEY = os.getenv("GROQ_API_KEY", "").strip()

if not BOT_TOKEN or not GROQ_KEY:
    raise RuntimeError("TELEGRAM_BOT_TOKEN and GROQ_API_KEY must be set in Environment Variables.")

bot = telebot.TeleBot(BOT_TOKEN, threaded=True)
ai = Groq(api_key=GROQ_KEY, timeout=30, max_retries=2)
memories = {}
mem_lock = threading.Lock()
ACTIVE_MODEL = None

# 2. Dynamic Model Resolver (404 Error Fix)
def get_available_model():
    """Finds the first available valid chat model from Groq API automatically."""
    global ACTIVE_MODEL
    if ACTIVE_MODEL:
        return ACTIVE_MODEL
    try:
        model_list = ai.models.list()
        available = [
            m.id for m in model_list.data 
            if m.id and not any(x in m.id.lower() for x in ('whisper', 'tts', 'guard', 'embed', 'transcribe'))
        ]
        log.info("Available Groq chat models: %s", available)
        
        # Priority fallback chain
        for preferred in (
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "llama3-70b-8192",
            "llama3-8b-8192",
            "mixtral-8x7b-32768"
        ):
            if preferred in available:
                ACTIVE_MODEL = preferred
                return ACTIVE_MODEL
        
        if available:
            ACTIVE_MODEL = available[0]
            return ACTIVE_MODEL
    except Exception as exc:
        log.warning("Could not auto-fetch model list: %s", exc)
    
    ACTIVE_MODEL = "llama-3.1-8b-instant"
    return ACTIVE_MODEL

# 3. Prompts
PERCEPTION_PROMPT = """Analyze the user's input and return ONLY a JSON object:
{
  "intent": "factual_query" | "conversational" | "deep_analysis",
  "domain": "Sports" | "Science_Tech" | "History" | "General",
  "lang": "bn" | "en"
}"""

SYSTEM_PROMPT = """You are Morning Intelligence, an autonomous factual AI assistant.
Rules:
1. Answer in the user's preferred language (Bengali or English).
2. Strict factual accuracy: Never invent winners, runners-up, dates, or scores.
3. Deliver high-impact, direct, and well-structured answers."""

CRITIC_PROMPT = """Audit the draft answer against the user query for factual errors or hallucinations (e.g., wrong winners/runners-up).
- If accurate, output ONLY: PASS
- If inaccurate, output the corrected version directly."""

# 4. Agent Functions
def perceive_input(text):
    try:
        model = get_available_model()
        res = ai.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": PERCEPTION_PROMPT}, {"role": "user", "content": text[:1000]}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return json.loads(res.choices[0].message.content)
    except Exception:
        return {"intent": "conversational", "domain": "General", "lang": "bn"}

def generate_actor_draft(chat_id, text):
    with mem_lock:
        history = list(memories.get(chat_id, [{"role": "system", "content": SYSTEM_PROMPT}]))
        history.append({"role": "user", "content": text})
        history = [history[0]] + history[1:][-8:]
        memories[chat_id] = history

    model = get_available_model()
    res = ai.chat.completions.create(
        model=model,
        messages=history,
        temperature=0.1,
        max_completion_tokens=1500
    )
    draft = re.sub(r"<think>.*?</think>", "", res.choices[0].message.content or "", flags=re.S).strip()
    return draft

def critic_evaluator(query, draft):
    try:
        model = get_available_model()
        audit_res = ai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CRITIC_PROMPT},
                {"role": "user", "content": f"Query: {query}\nDraft: {draft}"}
            ],
            temperature=0.0,
            max_completion_tokens=1500
        )
        verdict = audit_res.choices[0].message.content.strip()
        if verdict and not verdict.startswith("PASS"):
            return verdict
    except Exception as e:
        log.warning("Critic pass error: %s", e)
    return draft

# 5. Message Handlers
@bot.message_handler(commands=["start", "help"])
def start_cmd(message):
    bot.reply_to(
        message, 
        f"🤖 **Morning Intelligence Core লাইভ!**\nমডেল: `{get_available_model()}`\n\nযেকোনো প্রশ্ন করুন, উত্তর পাবেন যাচাইকৃত।"
    )

@bot.message_handler(commands=["clear"])
def clear_cmd(message):
    with mem_lock:
        memories[message.chat.id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    bot.reply_to(message, "🧹 মেমোরি ক্লিয়ার করা হয়েছে।")

@bot.message_handler(content_types=["text"])
def chat_handler(message):
    bot.send_chat_action(message.chat.id, "typing")
    text = (message.text or "").strip()

    # Step A: Perception
    _ = perceive_input(text)

    # Step B: Actor Generation
    draft = generate_actor_draft(message.chat.id, text)

    # Step C: Critic Audit
    final_output = critic_evaluator(text, draft)

    with mem_lock:
        memories[message.chat.id].append({"role": "assistant", "content": final_output[:3000]})

    bot.reply_to(message, final_output)

# 6. Render Health Server & Polling
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"Morning Intelligence Core is Healthy.")
    def log_message(self, *args): pass

def run_polling():
    time.sleep(2)
    try: bot.remove_webhook()
    except Exception: pass
    log.info("Polling loop running...")
    while True:
        try:
            bot.infinity_polling(timeout=20, long_polling_timeout=15, skip_pending=True)
        except Exception as e:
            log.error("Polling error: %s. Retrying in 5s...", e)
            time.sleep(5)

if __name__ == "__main__":
    threading.Thread(target=run_polling, daemon=True).start()
    port = int(os.getenv("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    server.serve_forever()
