import os
import logging
import threading
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, HTTPServer

from groq import AsyncGroq
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO
)
log = logging.getLogger("bot")

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = """You are an advanced, intelligent, and multi-functional AI Research & News Assistant inside a Telegram bot.

CORE ROLE
- Be a fast, polite, and precise assistant.
- Give clear, structured, factual answers: general knowledge, coding, maths, news analysis.
- Reply in Bengali if the user writes in Bengali, English if they write in English.

TELEGRAM FORMATTING
- Mobile-first: concise, scannable, short bullet points, no massive paragraphs.
- Put the most important answer or core data point in the VERY FIRST sentence. Details below.
- Use Markdown safely: **bold**, `inline code`, ``` code blocks ```. Always close every tag.
- Use emojis sparingly as section markers.

BOUNDARIES
- Stay professional, encouraging, and accurate.
- If you are unsure or lack real-time data, say so instead of guessing.
"""

client = AsyncGroq(api_key=GROQ_API_KEY)

# Per-chat memory: last 12 messages (6 exchanges)
history: dict[int, deque] = defaultdict(lambda: deque(maxlen=12))


# ---------- helpers ----------
def to_telegram_md(text: str) -> str:
    """Convert **bold** (standard MD) to *bold* (Telegram legacy Markdown)."""
    return text.replace("**", "*")


def split_text(text: str, limit: int = 4000):
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        yield text[:cut]
        text = text[cut:].lstrip("\n")
    if text:
        yield text


async def safe_reply(update: Update, text: str):
    for chunk in split_text(text):
        try:
            await update.message.reply_text(
                to_telegram_md(chunk), parse_mode=ParseMode.MARKDOWN
            )
        except BadRequest:
            # Broken Markdown from the LLM -> fall back to plain text
            await update.message.reply_text(chunk)


# ---------- handlers ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_reply(
        update,
        "👋 **Hello! I'm your AI Research & News Assistant.**\n\n"
        "• Ask me anything: news, coding, maths, general knowledge\n"
        "• বাংলায় লিখলে বাংলায় উত্তর দেব\n"
        "• `/clear` resets our conversation",
    )


async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    history.pop(update.effective_chat.id, None)
    await safe_reply(update, "🧹 Conversation cleared.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_text = update.message.text

    await context.bot.send_chat_action(chat_id, ChatAction.TYPING)
    history[chat_id].append({"role": "user", "content": user_text})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history[chat_id]]

    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=1024,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        log.exception("Groq error: %s", e)
        history[chat_id].pop()  # drop the failed user turn
        await update.message.reply_text(
            "⚠️ Something went wrong. Please try again in a moment.\n"
            "⚠️ কিছু সমস্যা হয়েছে, একটু পরে আবার চেষ্টা করুন।"
        )
        return

    history[chat_id].append({"role": "assistant", "content": answer})
    await safe_reply(update, answer)


async def handle_unsupported(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📩 I can only read text messages right now. Please type your question.\n"
        "📩 আমি এখন শুধু টেক্সট মেসেজ বুঝতে পারি। দয়া করে লিখে পাঠান।"
    )


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    log.error("Unhandled error", exc_info=context.error)


# ---------- keep-alive HTTP server (Render web services need an open port) ----------
class Health(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


def run_health_server():
    port = int(os.getenv("PORT", "10000"))
    HTTPServer(("0.0.0.0", port), Health).serve_forever()


# ---------- main ----------
def main():
    threading.Thread(target=run_health_server, daemon=True).start()

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("clear", clear))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(
        MessageHandler(~filters.TEXT & ~filters.COMMAND, handle_unsupported)
    )
    app.add_error_handler(on_error)

    log.info("Bot started with model %s", MODEL)
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
