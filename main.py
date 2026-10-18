import json
import logging
import os
import re
import threading
import time
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

# 2. Prompts
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

# 3. Agent Functions
def perceive_input(text):
    try:
        res = ai.chat.completions.create(
            model="llama-3.3-70b-versatile",
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

    res = ai.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=history,
        temperature=0.1,
        max_completion_tokens=1500
    )
    draft = re.sub(r"<think>.*?</think>", "", res.choices[0].message.content or "", flags=re.S).strip()
    return draft

def critic_evaluator(query, draft):
    try:
        audit_res = ai.chat.completions.create(
            model="llama-3.3-70b-versatile",
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

# 4. Message Handlers
@bot.message_handler(commands=["start", "help"])
def start_cmd(message):
    bot.reply_to(message, "🤖 **Morning Intelligence (Step 1: Perception & Anti-Hallucination Core) লাইভ!**\n\nযেকোনো প্রশ্ন করুন, উত্তর পাবেন নিখুঁত ও যাচাইকৃত।")

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

# 5. Render Health Server & Polling Thread
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"Step 1 Core is Healthy.")
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
