import os
import re
import time
import sqlite3
import threading
import requests
from http.server import BaseHTTPRequestHandler, HTTPServer
import telebot
from groq import Groq

# ==========================================
# 1. ENVIRONMENT VARIABLES
# ==========================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")

if not TELEGRAM_BOT_TOKEN or not GROQ_API_KEY:
    raise ValueError("Error: TELEGRAM_BOT_TOKEN and GROQ_API_KEY must be set in Environment Variables!")

# ==========================================
# 2. INITIALIZATION & SMART MODEL SELECTOR
# ==========================================
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
groq_client = Groq(api_key=GROQ_API_KEY)

user_memory = {}
user_mode = {}
AVAILABLE_MODELS = []

def refresh_available_models():
    global AVAILABLE_MODELS
    try:
        models = groq_client.models.list()
        AVAILABLE_MODELS = [
            m.id for m in models.data
            if not any(skip in m.id.lower() for skip in ["whisper", "guard", "tts", "embed", "vision"])
        ]
    except Exception as e:
        print(f"Error fetching models: {e}")

refresh_available_models()

def get_best_model(mode="fast"):
    if not AVAILABLE_MODELS:
        refresh_available_models()

    if mode == "deepseek":
        for m in AVAILABLE_MODELS:
            if "deepseek" in m.lower() or "qwen" in m.lower():
                return m

    preferred_fast = [
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "llama3-70b-8192",
        "llama-3.1-8b-instant",
        "llama3-8b-8192",
        "gemma2-9b-it",
        "mixtral-8x7b-32768"
    ]
    for pref in preferred_fast:
        if pref in AVAILABLE_MODELS:
            return pref

    return AVAILABLE_MODELS[0] if AVAILABLE_MODELS else "llama3-8b-8192"

# ==========================================
# 3. SYSTEM PROMPT & MEMORY HELPERS
# ==========================================
SYSTEM_PROMPT = """You are an advanced, intelligent, and multi-functional AI Research & News Assistant integrated into a Telegram Bot. Your intelligence is powered by the Groq API, and you run asynchronously on a Render cloud server.

### 1. CORE ROLE & CAPABILITIES
- Act as a fast, polite, and precise AI assistant.
- Provide clear, well-structured, and factual answers to user queries.
- Automatically adapt to the user's language (respond in Bengali if the user writes in Bengali, and English if they write in English).

### 2. TELEGRAM STYLING & FORMATTING GUIDELINES
- MOBILE OPTIMIZATION: Keep responses concise and scannable. Use short bullet points.
- DIRECT HIERARCHY: Always place the most important answer in the very first sentence.
- INTERACTIVE MARKDOWN: Use standard Markdown (**bold**, `inline code`, and ```code blocks```) safely."""

def build_safe_messages(chat_id, new_text):
    if len(new_text) > 2500:
        new_text = new_text[:2500] + "\n...[Text trimmed]"

    if chat_id not in user_memory or not user_memory[chat_id]:
        user_memory[chat_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    user_memory[chat_id].append({"role": "user", "content": new_text})

    if len(user_memory[chat_id]) > 5:
        user_memory[chat_id] = [user_memory[chat_id][0]] + user_memory[chat_id][-4:]

    while len(user_memory[chat_id]) > 2:
        total_chars = sum(len(m["content"]) for m in user_memory[chat_id])
        if total_chars > 5000:
            user_memory[chat_id].pop(1)
        else:
            break

    return user_memory[chat_id]

def clean_deepseek_output(text):
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return cleaned if cleaned else text

def send_smart_reply(message, text):
    max_len = 4000
    for i in range(0, len(text), max_len):
        chunk = text[i:i + max_len]
        try:
            bot.reply_to(message, chunk, parse_mode="Markdown", disable_web_page_preview=True)
        except Exception:
            bot.reply_to(message, chunk, disable_web_page_preview=True)

# ==========================================
# 4. PHASE 2: NEWS API & SQLITE DATABASE
# ==========================================
def init_db():
    try:
        conn = sqlite3.connect('ai_research.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_store (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT,
                title TEXT,
                description TEXT,
                url TEXT,
                verified INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database init error: {e}")

init_db()

def fetch_news_module(topic):
    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key:
        return (
            "⚠️ **NEWS_API_KEY পাওয়া যায়নি!**\n\n"
            "খবর খোঁজার জন্য:\n"
            "১. `newsapi.org` থেকে ফ্রি API Key নিন।\n"
            "২. Render-এর Environment Variables-এ গিয়ে `NEWS_API_KEY` নামে সেভ করুন।"
        )

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": topic,
        "sortBy": "publishedAt",
        "pageSize": 5,
        "language": "en",
        "apiKey": api_key
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if data.get("status") != "ok":
            return f"⚠️ News API Error: {data.get('message', 'Unknown error')}"

        articles = data.get("articles", [])
        if not articles:
            return f"🔍 `{topic}` বিষয়ে কোনো নতুন খবর পাওয়া যায়নি।"

        conn = sqlite3.connect('ai_research.db')
        cursor = conn.cursor()

        news_text = f"📰 **'{topic}' বিষয়ক সর্বশেষ খবর (SQL ডাটাবেসে সেভ করা হয়েছে):**\n\n"

        for idx, art in enumerate(articles, 1):
            title = (art.get("title") or "No Title").replace("*", "").replace("_", "").replace("`", "")
            desc = (art.get("description") or "বিস্তারিত লিংকে দেখুন...").replace("*", "").replace("_", "").replace("`", "")
            link = art.get("url") or "https://news.google.com"

            cursor.execute(
                "INSERT INTO news_store (topic, title, description, url) VALUES (?, ?, ?, ?)",
                (topic, title, desc, link)
            )

            news_text += f"🔹 **{idx}. {title}**\n{desc}\n🔗 [সম্পূর্ণ খবর পড়ুন]({link})\n\n"

        conn.commit()
        conn.close()
        return news_text

    except Exception as e:
        return f"⚠️ খবর আনতে সমস্যা হয়েছে: {str(e)}"

def get_saved_news_from_sql():
    try:
        conn = sqlite3.connect('ai_research.db')
        cursor = conn.cursor()
        cursor.execute("SELECT topic, title, url, created_at FROM news_store ORDER BY id DESC LIMIT 5")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return "📭 SQL ডাটাবেসে এখনো কোনো খবর সেভ করা হয়নি। প্রথমে `/news AI` লিখে সার্চ করুন!"

        text = "🗄️ **SQL ডাটাবেসে সেভ থাকা সর্বশেষ ৫টি খবর:**\n\n"
        for idx, (topic, title, url, created_at) in enumerate(rows, 1):
            text += f"{idx}. **[{topic.upper()}]** {title}\n🔗 [Link]({url}) | 🕒 `{created_at}`\n\n"
        return text
    except Exception as e:
        return f"⚠️ ডাটাবেস রিড এরর: {str(e)}"

# ==========================================
# 5. FUTURE PIPELINE MODULES (Phase 3-6 Hooks)
# ==========================================
def verify_with_deepseek():
    pass

def format_with_gemini():
    pass

def generate_pdf_hf():
    pass

def generate_audio_elevenlabs():
    pass

# ==========================================
# 6. TELEGRAM BOT COMMANDS & HANDLERS
# ==========================================
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    chat_id = message.chat.id
    mode = user_mode.get(chat_id, "fast")
    active_model = get_best_model(mode)
    
    welcome_text = (
        f"⚡ **Groq AI & News Assistant is Online!**\n\n"
        f"🧠 **Active Engine:** `{active_model}`\n\n"
        "📌 **Commands:**\n"
        "• `/news <topic>` - লেটেস্ট খবর খুঁজুন ও SQL-এ সেভ করুন\n"
        "• `/saved` - SQL ডাটাবেসে সেভ থাকা খবর দেখুন\n"
        "• `/fast` - দ্রুত চ্যাট মোড (Llama 3)\n"
        "• `/deepseek` - ডিপ থিংকিং মোড (DeepSeek)\n"
        "• `/clear` - চ্যাট মেমোরি ক্লিয়ার করুন\n"
        "• `/model` - বর্তমান মডেল দেখুন"
    )
    send_smart_reply(message, welcome_text)

@bot.message_handler(commands=['news'])
def handle_news_command(message):
    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        bot.reply_to(
            message,
            "⚠️ অনুগ্রহ করে `/news` এর পরে কোনো বিষয় লিখুন।\nযেমন: `/news Artificial Intelligence`",
            parse_mode="Markdown"
        )
        return
    
    topic = args[1].strip()
    bot.send_chat_action(message.chat.id, 'typing')
    result = fetch_news_module(topic)
    send_smart_reply(message, result)

@bot.message_handler(commands=['saved'])
def handle_saved_news(message):
    bot.send_chat_action(message.chat.id, 'typing')
    result = get_saved_news_from_sql()
    send_smart_reply(message, result)

@bot.message_handler(commands=['deepseek'])
def set_deepseek_mode(message):
    chat_id = message.chat.id
    user_mode[chat_id] = "deepseek"
    model = get_best_model("deepseek")
    bot.reply_to(message, f"🧠 **DeepSeek Mode Activated!**\nModel: `{model}`", parse_mode="Markdown")

@bot.message_handler(commands=['fast'])
def set_fast_mode(message):
    chat_id = message.chat.id
    user_mode[chat_id] = "fast"
    model = get_best_model("fast")
    bot.reply_to(message, f"⚡ **Fast Mode Activated!**\nModel: `{model}`", parse_mode="Markdown")

@bot.message_handler(commands=['model'])
def show_current_model(message):
    chat_id = message.chat.id
    mode = user_mode.get(chat_id, "fast")
    model = get_best_model(mode)
    bot.reply_to(message, f"📌 **Mode:** `{mode.upper()}` | 🧠 **Model:** `{model}`", parse_mode="Markdown")

@bot.message_handler(commands=['clear'])
def clear_chat_memory(message):
    chat_id = message.chat.id
    user_memory[chat_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    bot.reply_to(message, "🧹 **Memory cleared!**", parse_mode="Markdown")

@bot.message_handler(content_types=['audio', 'photo', 'voice', 'video', 'document', 'sticker', 'location', 'contact'])
def handle_unsupported_media(message):
    bot.reply_to(message, "⚠️ Please send text messages only! *(অনুগ্রহ করে টেক্সট পাঠান)*", parse_mode="Markdown")

@bot.message_handler(content_types=['text'])
def handle_all_messages(message):
    chat_id = message.chat.id
    user_text = message.text

    if not user_text:
        return

    bot.send_chat_action(chat_id, 'typing')
    
    mode = user_mode.get(chat_id, "fast")
    current_model = get_best_model(mode)
    safe_messages = build_safe_messages(chat_id, user_text)

    try:
        response = groq_client.chat.completions.create(
            model=current_model,
            messages=safe_messages,
            temperature=0.7,
            max_tokens=2000,
            stream=True
        )

        full_reply = ""
        for chunk in response:
            content = chunk.choices[0].delta.content or ""
            full_reply += content

        full_reply = clean_deepseek_output(full_reply)
        user_memory[chat_id].append({"role": "assistant", "content": full_reply[:1200]})
        
        send_smart_reply(message, full_reply)

    except Exception as e:
        print(f"Streaming error ({current_model}): {e}")
        try:
            refresh_available_models()
            fallback_model = get_best_model("fast")
            
            emergency_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text[:1000]}
            ]
            user_memory[chat_id] = emergency_messages

            completion = groq_client.chat.completions.create(
                model=fallback_model,
                messages=emergency_messages,
                temperature=0.7,
                max_tokens=1024
            )
            ai_reply = clean_deepseek_output(completion.choices[0].message.content)
            user_memory[chat_id].append({"role": "assistant", "content": ai_reply[:1000]})
            send_smart_reply(message, ai_reply)
        except Exception:
            bot.reply_to(message, "⚠️ Server busy. Type `/clear` to reset.")

# ==========================================
# 7. RENDER HEALTH CHECK SERVER
# ==========================================
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write("AI Bot is running 24/7 on Render!".encode("utf-8"))

    def log_message(self, format, *args):
        return

def run_dummy_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(('0.0.0.0', port), HealthCheckHandler)
    server.serve_forever()

# ==========================================
# 8. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    server_thread = threading.Thread(target=run_dummy_server)
    server_thread.daemon = True
    server_thread.start()

    print("Starting Telegram AI Bot...")
    try:
        bot.remove_webhook()
        time.sleep(2)
    except Exception as e:
        print(f"Webhook warning: {e}")

    bot.infinity_polling(timeout=20, long_polling_timeout=10, skip_pending=True)
    
