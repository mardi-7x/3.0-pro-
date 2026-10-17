import os
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
    raise ValueError("Error: TELEGRAM_BOT_TOKEN এবং GROQ_API_KEY এনভায়রনমেন্ট ভেরিয়েবলে সেট করা নেই!")

# ==========================================
# 2. INITIALIZATION & AUTO MODEL SELECTOR
# ==========================================
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
groq_client = Groq(api_key=GROQ_API_KEY)

user_memory = {}
ACTIVE_MODEL = None

def get_working_model():
    """Groq API থেকে স্বয়ংক্রিয়ভাবে সচল চ্যাট মডেল খুঁজে বের করার ফাংশন"""
    global ACTIVE_MODEL
    if ACTIVE_MODEL:
        return ACTIVE_MODEL
    
    try:
        models = groq_client.models.list()
        available_ids = [m.id for m in models.data]
        print("Available Groq Models:", available_ids)
        
        # পছন্দের মডেলের তালিকা (যেটি আগে পাওয়া যাবে সেটি সেট হবে)
        preferred_models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "llama3-70b-8192",
            "llama3-8b-8192",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ]
        
        for pref in preferred_models:
            if pref in available_ids:
                ACTIVE_MODEL = pref
                print(f"Selected Preferred Model: {ACTIVE_MODEL}")
                return ACTIVE_MODEL
        
        # ওপরের কোনোটিই না থাকলে অডিও (whisper) বা গার্ড মডেল বাদে যেকোনো সচল চ্যাট মডেল নেবে
        for m_id in available_ids:
            if not any(skip in m_id.lower() for skip in ["whisper", "guard", "tts", "embed", "vision"]):
                ACTIVE_MODEL = m_id
                print(f"Selected Fallback Model: {ACTIVE_MODEL}")
                return ACTIVE_MODEL
                
    except Exception as e:
        print(f"Error fetching model list: {e}")
    
    # ডিফল্ট ফলব্যাক
    return "llama3-8b-8192"

# শুরুতেই সচল মডেলটি খুঁজে নেওয়া হচ্ছে
ACTIVE_MODEL = get_working_model()

# ==========================================
# 3. FUTURE PIPELINE MODULES (পরবর্তী ধাপগুলোর জন্য প্রস্তুত)
# ==========================================
def fetch_news_module(topic):
    """ধাপ ২: News API দিয়ে খবর খোঁজা"""
    pass

def verify_and_store_sql(news_data):
    """ধাপ ৩: DeepSeek API দিয়ে ভেরিফাই এবং SQL ডাটাবেসে সেভ করা"""
    pass

def format_with_gemini(verified_data):
    """ধাপ ৪: Gemini API দিয়ে সুন্দর রিপোর্ট ফরম্যাট তৈরি"""
    pass

def generate_pdf_hf(formatted_text):
    """ধাপ ৫: Hugging Face API দিয়ে PDF তৈরি"""
    pass

def generate_audio_elevenlabs(summary_text):
    """ধাপ ৬: ElevenLabs API দিয়ে অডিও ওভারভিউ তৈরি"""
    pass

# ==========================================
# 4. TELEGRAM BOT HANDLERS
# ==========================================
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    model_name = get_working_model()
    welcome_text = (
        "হ্যালো! আমি আপনার এআই রিসার্চ ও নিউজ অ্যাসিস্ট্যান্ট।\n\n"
        f"বর্তমানে আমি **Groq API ({model_name})** এর মাধ্যমে সুপার-ফাস্ট চ্যাটবট হিসেবে কাজ করছি।\n"
        "যেকোনো প্রশ্ন করুন, আমি সাথে সাথে উত্তর দেবো!\n\n"
        "*(মেমোরি রিসেট করতে /clear এবং সচল মডেল দেখতে /model লিখুন)*"
    )
    bot.reply_to(message, welcome_text, parse_mode="Markdown")

@bot.message_handler(commands=['model'])
def show_model(message):
    model_name = get_working_model()
    bot.reply_to(message, f"বর্তমানে সচল থাকা এআই মডেল: `{model_name}`", parse_mode="Markdown")

@bot.message_handler(commands=['clear'])
def clear_memory(message):
    chat_id = message.chat.id
    user_memory[chat_id] = []
    bot.reply_to(message, "মেমোরি ক্লিয়ার করা হয়েছে! নতুন করে কথা শুরু করতে পারেন।")

@bot.message_handler(func=lambda message: True)
def handle_chat(message):
    global ACTIVE_MODEL
    chat_id = message.chat.id
    user_text = message.text

    if not user_text:
        return

    # চ্যাট হিস্ট্রি সেটআপ
    if chat_id not in user_memory or not user_memory[chat_id]:
        user_memory[chat_id] = [
            {
                "role": "system",
                "content": "You are a helpful, smart, and fast AI assistant. You reply clearly in Bengali or English depending on the user's language."
            }
        ]

    # ইউজারের মেসেজ মেমোরিতে যোগ করা
    user_memory[chat_id].append({"role": "user", "content": user_text})

    # মেমোরি যাতে অতিরিক্ত বড় না হয় (শেষ ১০টি মেসেজ রাখা হবে)
    if len(user_memory[chat_id]) > 11:
        user_memory[chat_id] = [user_memory[chat_id][0]] + user_memory[chat_id][-10:]

    try:
        bot.send_chat_action(chat_id, 'typing')
        current_model = get_working_model()

        # Groq API কল
        completion = groq_client.chat.completions.create(
            model=current_model,
            messages=user_memory[chat_id],
            temperature=0.7,
            max_tokens=1024,
        )

        ai_reply = completion.choices[0].message.content

        # AI-এর উত্তর মেমোরিতে সেভ করা এবং টেলিগ্রামে পাঠানো
        user_memory[chat_id].append({"role": "assistant", "content": ai_reply})
        bot.reply_to(message, ai_reply)

    except Exception as e:
        # যদি কোনো কারণে মডেল পরিবর্তন হয়, তবে রিসেট করে নতুন সচল মডেল দিয়ে আবার চেষ্টা করবে
        ACTIVE_MODEL = None
        fallback_model = get_working_model()
        try:
            completion = groq_client.chat.completions.create(
                model=fallback_model,
                messages=user_memory[chat_id],
                temperature=0.7,
                max_tokens=1024,
            )
            ai_reply = completion.choices[0].message.content
            user_memory[chat_id].append({"role": "assistant", "content": ai_reply})
            bot.reply_to(message, ai_reply)
        except Exception as retry_error:
            bot.reply_to(message, f"একটি টেকনিক্যাল সমস্যা হয়েছে: {str(retry_error)}")

# ==========================================
# 5. RENDER HEALTH CHECK SERVER (Port Binding)
# ==========================================
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write("Bot is running smoothly on Render!".encode("utf-8"))

    def log_message(self, format, *args):
        return

def run_dummy_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(('0.0.0.0', port), HealthCheckHandler)
    print(f"Health check server listening on port {port}")
    server.serve_forever()

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # ব্যাকগ্রাউন্ডে ডামি ওয়েব সার্ভার চালু করা
    server_thread = threading.Thread(target=run_dummy_server)
    server_thread.daemon = True
    server_thread.start()

    print(f"Telegram Bot is polling using model: {ACTIVE_MODEL}...")
    bot.infinity_polling(timeout=20, long_polling_timeout=10)
    
