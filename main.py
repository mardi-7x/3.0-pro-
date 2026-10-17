
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import telebot
from groq import Groq

# ==========================================
# 1. ENVIRONMENT VARIABLES (Render থেকে লোড হবে)
# ==========================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not TELEGRAM_BOT_TOKEN or not GROQ_API_KEY:
    raise ValueError("Error: TELEGRAM_BOT_TOKEN এবং GROQ_API_KEY এনভায়রনমেন্ট ভেরিয়েবলে সেট করা নেই!")

# ==========================================
# 2. INITIALIZATION (Bot & Groq Client)
# ==========================================
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
groq_client = Groq(api_key=GROQ_API_KEY)

# ইউজারদের কথোপকথন মনে রাখার জন্য মেমোরি (পরবর্তীতে এটি SQL-এ যাবে)
user_memory = {}

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
    """ধাপ ৫: Hugging Face API/Library দিয়ে PDF তৈরি"""
    pass

def generate_audio_elevenlabs(summary_text):
    """ধাপ ৬: ElevenLabs API দিয়ে অডিও ওভারভিউ তৈরি"""
    pass

# ==========================================
# 4. TELEGRAM BOT HANDLERS (ধাপ ১: Groq Fast Chatbot)
# ==========================================
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    welcome_text = (
        "হ্যালো! আমি আপনার এআই রিসার্চ ও নিউজ অ্যাসিস্ট্যান্ট।\n\n"
        "বর্তমানে আমি **Groq API**-এর মাধ্যমে সুপার-ফাস্ট চ্যাটবট হিসেবে কাজ করছি।\n"
        "যেকোনো প্রশ্ন করুন, আমি সাথে সাথে উত্তর দেবো!\n\n"
        "*(রিসেট করতে /clear লিখুন)*"
    )
    bot.reply_to(message, welcome_text, parse_mode="Markdown")

@bot.message_handler(commands=['clear'])
def clear_memory(message):
    chat_id = message.chat.id
    user_memory[chat_id] = []
    bot.reply_to(message, "মেমোরি ক্লিয়ার করা হয়েছে! নতুন করে কথা শুরু করতে পারেন।")

@bot.message_handler(func=lambda message: True)
def handle_chat(message):
    chat_id = message.chat.id
    user_text = message.text

    # চ্যাট হিস্ট্রি সেটআপ
    if chat_id not in user_memory:
        user_memory[chat_id] = [
            {
                "role": "system",
                "content": "You are a helpful, smart, and fast AI assistant. You reply clearly in Bengali or English depending on the user's language."
            }
        ]

    # ইউজারের মেসেজ মেমোরিতে যোগ করা
    user_memory[chat_id].append({"role": "user", "content": user_text})

    # মেমোরি যাতে অতিরিক্ত বড় না হয়, তাই শেষ ১০টি মেসেজ রাখা হচ্ছে
    if len(user_memory[chat_id]) > 11:
        user_memory[chat_id] = [user_memory[chat_id][0]] + user_memory[chat_id][-10:]

    try:
        # টেলিগ্রামে 'Typing...' স্ট্যাটাস দেখানো
        bot.send_chat_action(chat_id, 'typing')

        # Groq API কল (Llama 3.3 70B মডেল ব্যবহার করা হয়েছে দ্রুত ও নিখুঁত উত্তরের জন্য)
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=user_memory[chat_id],
            temperature=0.7,
            max_tokens=1024,
        )

        ai_reply = completion.choices[0].message.content

        # AI-এর উত্তর মেমোরিতে সেভ করা এবং টেলিগ্রামে পাঠানো
        user_memory[chat_id].append({"role": "assistant", "content": ai_reply})
        bot.reply_to(message, ai_reply)

    except Exception as e:
        bot.reply_to(message, f"একটি টেকনিক্যাল সমস্যা হয়েছে: {str(e)}")

# ==========================================
# 5. RENDER HEALTH CHECK SERVER (Port Binding)
# ==========================================
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write("Bot is running smoothly on Render!".encode("utf-8"))

    # অপ্রয়োজনীয় লগ বন্ধ রাখার জন্য
    def log_message(self, format, *args):
        return

def run_Dummy_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(('0.0.0.0', port), HealthCheckHandler)
    print(f"Health check server listening on port {port}")
    server.serve_forever()

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # ব্যাকগ্রাউন্ডে ডামি ওয়েব সার্ভার চালু করা (Render-এর জন্য আবশ্যিক)
    server_thread = threading.Thread(target=run_Dummy_server)
    server_thread.daemon = True
    server_thread.start()

    print("Telegram Bot is polling...")
    # টেলিগ্রাম বট চালু করা
    bot.infinity_polling(timeout=20, long_polling_timeout=10)
    
