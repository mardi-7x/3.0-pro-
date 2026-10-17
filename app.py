
import os
import json
import logging
import asyncio
import requests

from openai import OpenAI
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ---------- CONFIGURATION ----------

BOT_TOKEN = os.getenv("BOT_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")

BASE_URL = "https://api.deepseek.com"
SEARCH_URL = "https://api.deepseek.com/v1/tools/search"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("MorningIntelligence")

if not BOT_TOKEN or not DEEPSEEK_API_KEY:
    raise RuntimeError(
        "Set BOT_TOKEN and DEEPSEEK_API_KEY "
        "in your environment variables."
    )

ai = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=BASE_URL,
    timeout=60.0,
    max_retries=2,
)

SYSTEM_PROMPT = """
You are Morning Intelligence 2.0, an AI research assistant.

Rules:
- Answer clearly and accurately.
- Respect the user's language.
- Never invent news, sources, dates, or quotations.
- For news, use the supplied search results as evidence.
- Include source URLs when available.
- Distinguish confirmed facts from uncertain claims.
- If evidence is insufficient, say so honestly.
- Do not claim that you searched the web unless search was used.
"""


# ---------- WEB SEARCH ----------

def search_web(query: str) -> list:
    response = requests.post(
        SEARCH_URL,
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "query": query,
            "count": 8,
        },
        timeout=30,
    )

    response.raise_for_status()
    data = response.json()

    results = data.get("results", [])

    # Keep only results with usable source URLs.
    clean_results = []

    for item in results:
        url = item.get("url", "")
        if not url.startswith(("https://", "http://")):
            continue

        clean_results.append({
            "title": item.get("title", "Untitled"),
            "url": url,
            "snippet": item.get("snippet", ""),
        })

    return clean_results


# ---------- DEEPSEEK AI ----------

def generate_answer(prompt: str, search: bool = False) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    if search:
        results = search_web(prompt)

        if not results:
            return (
                "No usable search results were returned. "
                "No news summary was generated."
            )

        evidence = json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        )

        messages.append({
            "role": "user",
            "content": (
                "Research this topic using ONLY the supplied "
                "search results as your initial evidence:\n\n"
                f"TOPIC:\n{prompt}\n\n"
                f"SEARCH RESULTS:\n{evidence}\n\n"
                "Summarize the most relevant findings. "
                "Include the original source URLs. "
                "Do not treat snippets alone as conclusive proof. "
                "If the results do not establish a claim, "
                "state that limitation clearly."
            ),
        })
    else:
        messages.append({
            "role": "user",
            "content": prompt,
        })

    response = ai.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=2500,
    )

    answer = response.choices[0].message.content

    return answer or "I couldn't generate a response."


async def ask_ai(prompt: str, search: bool = False) -> str:
    return await asyncio.to_thread(
        generate_answer,
        prompt,
        search,
    )


# ---------- TELEGRAM HELPERS ----------

async def send_long_message(message, text: str):
    # Keep each Telegram message below its length limit.
    chunks = [
        text[i:i + 3900]
        for i in range(0, len(text), 3900)
    ]

    if not chunks:
        chunks = ["No response was generated."]

    await message.reply_text(chunks[0])

    for chunk in chunks[1:]:
        await message.reply_text(chunk)


# ---------- COMMANDS ----------

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    await update.effective_message.reply_text(
        "🌅 Welcome to Morning Intelligence 2.0!\n\n"
        "Powered by DeepSeek AI.\n\n"
        "💬 Ask me a question.\n"
        "🔎 /news AI and space\n"
        "❓ /help"
    )


async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    await update.effective_message.reply_text(
        "🌍 Morning Intelligence 2.0\n\n"
        "• Ask general questions\n"
        "• /news AI and space\n"
        "• /news robotics\n"
        "• /news semiconductor industry\n\n"
        "News research requires access to the "
        "DeepSeek Search API."
    )


async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    message = update.effective_message

    if not message or not message.text:
        return

    question = message.text.strip()
    status = await message.reply_text("🧠 Thinking...")

    try:
        answer = await ask_ai(question)

        await status.delete()
        await send_long_message(message, answer)

    except Exception:
        log.exception("Chat request failed")

        await status.edit_text(
            "⚠️ I couldn't complete your request. "
            "Please check the API configuration and try again."
        )


async def news_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    message = update.effective_message
    topic = " ".join(context.args).strip()

    if not topic:
        topic = "latest AI and technology news"

    status = await message.reply_text(
        "🔎 Searching for news sources..."
    )

    try:
        answer = await ask_ai(
            "Find recent news about: " + topic,
            search=True,
        )

        await status.delete()
        await send_long_message(message, answer)

    except Exception:
        log.exception("News research failed")

        await status.edit_text(
            "⚠️ News research failed.\n\n"
            "Check your DeepSeek API key, account access, "
            "available balance and Search API response."
        )


# ---------- START BOT ----------

def main():
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("news", news_command))

    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message,
        )
    )

    log.info("Morning Intelligence 2.0 is starting...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
