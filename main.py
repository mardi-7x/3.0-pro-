import os
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import telebot
from groq import Groq

# ==========================================
# 1. ENVIRONMENT VARIABLES
# ==========================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

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
    """Fetches the live list of working models from your Groq account."""
    global AVAILABLE_MODELS
    try:
        models = groq_client.models.list()
        AVAILABLE_MODELS = [
            m.id for m in models.data
            if not any(skip in m.id.lower() for skip in ["whisper", "guard", "tts", "embed", "vision"])
        ]
        print("Live Groq Models:", AVAILABLE_MODELS)
    except Exception as e:
        print(f"Error fetching models: {e}")

refresh_available_models()

def get_best_model(mode="fast"):
    """Selects the best working model based on user mode (DeepSeek vs Fast Chat)."""
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
# 3. YOUR CUSTOM SYSTEM PROMPT & MEMORY HELPERS
# ==========================================
SYSTEM_PROMPT = """You are a helpful, direct, and fast AI assistant integrated into a Telegram Bot. Your intelligence is powered by the Groq API, and you are running asynchronously on a Render cloud server.

### 1. ENVIRONMENT & CONTEXT
- PLATFORM: Telegram Messenger. Your responses must be optimized for mobile reading.
- INFRASTRUCTURE: Hosted on Render web services. Optimize performance for low-latency delivery.
- CORE ENGINE: Groq API (LLM). You process user messages and return instantaneous, high-quality responses.

### 2. RESPONSE GUIDELINES (TELEGRAM STYLING)
- INTERACTIVE FORMATTING: Use standard Markdown (**bold**, `inline code`, and ```code blocks```). Always ensure tags are strictly closed to prevent Telegram parsing errors.
- BREVITY & SCANNABILITY: Keep sentences short and use active voice. Users on Telegram read on mobile screens; avoid long paragraphs. Use functional emojis only as visual bullets or section breaks.
- DIRECT HIERARCHY: Always put the most critical answer, data point, or response in the very first sentence. Provide background details or context below it.
- LANGUAGE: Reply in the same language the user writes in (Bengali or English).

### 3. OPERATIONAL BOUNDARIES
- Maintain strict chat context memory up to the application's configuration limits.
- Remain polite, engaging, and professional. Match the user's tone while maintaining a peer-like, helpful demeanor."""

def build_safe_messages(chat_id, new_text):
    """Strictly controls token length so 'Error 400: context_length_exceeded' never happens."""
    if len(new_text) > 2500:
        new_text = new_text[:2500] + "\n...[Text trimmed to fit memory limit]"

    if chat_id not in user_memory or not user_memory[chat_id]:
        user_memory[chat_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    user_memory[chat_id].append({"role": "user", "content": new_text})

    # Keep system prompt + last 4 messages
    if len(user_memory[chat_id]) > 5:
        user_memory[chat_id] = [user_memory[chat_id][0]] + user_memory[chat_id][-4:]

    # Cap total memory characters at 5000
    while len(user_memory[chat_id]) > 2:
        total_chars = sum(len(m["content"]) for m in user_memory[chat_id])
        if total_chars > 5000:
            user_memory[chat_id].pop(1)
        else:
            break

    return user_memory[chat_id]

def clean_deepseek_output(text):
    """Removes <think>...</think> tags from DeepSeek R1 outputs."""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return cleaned if cleaned else text

def send_smart_reply(message, text):
    """Splits long messages into chunks and safely handles Telegram formatting."""
    max_len = 4000
    for i in range(0, len(text), max_len):
        chunk = text[i:i + max_len]
        try:
            bot.reply_to(message, chunk, parse_mode="Markdown")
        except Exception:
            # Automatic fallback to plain text if unclosed Markdown symbols exist
            bot.reply_to(message, chunk)

# ==========================================
# 4. FUTURE PIPELINE MODULES (Phase 2-6 Hooks)
# ==========================================
def fetch_news_module(topic):
    pass

def verify_and_store_sql(news_data):
    pass

def format_with_gemini(verified_data):
    pass

def generate_pdf_hf(formatted_text):
    pass

def generate_audio_elevenlabs(summary_text):
    pass

# ==========================================
# 5. TELEGRAM BOT COMMANDS & HANDLERS
# ==========================================
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    chat_id = message.chat.id
    mode = user_mode.get(chat_id, "fast")
    active_model = get_best_model(mode)
    
    welcome_text = (
        "⚡ **Groq AI Assistant is Online!**\n\n"
        f"🧠 **Active Engine:** `{active_model}`\n\n"
        "📌 **Quick Commands:**\n"
        "• `/fast` - Instant replies (Llama 3)\n"
        "• `/deepseek` - Deep reasoning mode (DeepSeek)\n"
        "• `/clear` - Reset conversation memory\n"
        "• `/model` - Check active model\n\n"
        "Send any text question in **Bengali** or **English** to begin!"
    )
    send_smart_reply(message, welcome_text)

@bot.message_handler(commands=['deepseek'])
def set_deepseek_mode(message):
    chat_id = message.chat.id
    user_mode[chat_id] = "deepseek"
    model = get_best_model("deepseek")
    bot.reply_to(message, f"🧠 **DeepSeek Reasoning Mode Activated!**\nActive Model: `{model}`", parse_mode="Markdown")

@bot.message_handler(commands=['fast'])
def set_fast_mode(message):
    chat_id = message.chat.id
    user_mode[chat_id] = "fast"
    model = get_best_model("fast")
    bot.reply_to(message, f"⚡ **Fast Mode Activated!**\nActive Model: `{model}`", parse_mode="Markdown")

@bot.message_handler(commands=['model'])
def show_current_model(message):
    chat_id = message.chat.id
    mode = user_mode.get(chat_id, "fast")
    model = get_best_model(mode)
    bot.reply_to(message, f"📌 **Current Mode:** `{mode.upper()}`\n🧠 **Active Model:** `{model}`", parse_mode="Markdown")

@bot.message_handler(commands=['clear'])
def clear_chat_memory(message):
    chat_id = message.chat.id
    user_memory[chat_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    bot.reply_to(message, "🧹 **Memory cleared!** Send a new message to start fresh.", parse_mode="Markdown")

# Operational Boundary: Handle unsupported media types (voice, photos, documents, etc.)
@bot.message_handler(content_types=['audio', 'photo', 'voice', 'video', 'document', 'sticker', 'location', 'contact'])
def handle_unsupported_media(message):
    reply = (
        "⚠️ **Text Messages Only**\n"
        "I currently process text input only. Please type your question or paste your text directly! "
        "*(টেলিগ্রামে অনুগ্রহ করে টেক্সট মেসেজ পাঠান)*"
    )
    bot.reply_to(message, reply, parse_mode="Markdown")

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
        completion = groq_client.chat.completions.create(
            model=current_model,
            messages=safe_messages,
            temperature=0.6,
            max_tokens=1024,
        )
        ai_reply = completion.choices[0].message.content
        ai_reply = clean_deepseek_output(ai_reply)

        user_memory[chat_id].append({"role": "assistant", "content": ai_reply[:1200]})
        send_smart_reply(message, ai_reply)

    except Exception as e:
        print(f"First attempt error ({current_model}): {e}")
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
                temperature=0.6,
                max_tokens=800,
            )
            ai_reply = completion.choices[0].message.content
            ai_reply = clean_deepseek_output(ai_reply)

            user_memory[chat_id].append({"role": "assistant", "content": ai_reply[:1000]})
            send_smart_reply(message, ai_reply)

        except Exception as final_error:
            bot.reply_to(message, f"⚠️ Server busy. Please type `/clear` and try again.\nDetails: {str(final_error)[:150]}")

# ==========================================
# 6. RENDER HEALTH CHECK SERVER (24/7 Keep-Alive)
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
    print(f"Health check server listening on port {port}")
    server.serve_forever()

# ==========================================
# 7. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    server_thread = threading.Thread(target=run_dummy_server)
    server_thread.daemon = True
    server_thread.start()

    print("Telegram AI Bot is polling...")
    bot.infinity_polling(timeout=20, long_polling_timeout=10)
    
