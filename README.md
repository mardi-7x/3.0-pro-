# Morning Intelligence 2.0 — Python/Render edition

## Files
- `app.py`: Telegram polling bot, NewsAPI collection, GROQ evidence review, PDF generation, daily scheduler.
- `requirements.txt`: Python dependencies.
- `render.yaml`: optional Render Blueprint configuration.
- `Procfile`: one-worker Gunicorn start command.

## Required Render environment variables
- `BOT_TOKEN`: token from Telegram BotFather.
- `GROQ_API_KEY`: API key from GroqCloud.
- `NEWS_API_KEY`: NewsAPI key. Optional alternative: `NEWS_API_KEY_2`.
- `HF_TOKEN`: Hugging Face token with Inference Providers permission; required for speech-to-text.

Optional:
- `GROQ_MODEL`: defaults to `openai/gpt-oss-120b`; set to a model currently enabled in your Groq account.
- `HF_ASR_PROVIDER`: defaults to `fal-ai` for Whisper speech recognition.
- `ASR_MODEL`: defaults to `openai/whisper-large-v3`.
- `TIMEZONE`: defaults to `Asia/Kolkata`.
- `DAILY_HOUR`: defaults to `9`.
- `DAILY_MINUTE`: defaults to `30`.
- `MAX_ARTICLES`: defaults to `6`.
- `LOOKBACK_HOURS`: defaults to `24`.
- `DATA_DIR`: defaults to `data`; for persistent subscriber/history storage, mount a Render persistent disk and set this to its mount path, e.g. `/var/data`.

## Deploy
1. Push these files to a GitHub repository.
2. In Render, create a **Web Service** connected to the repository.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app --workers 1 --threads 4 --timeout 120`
5. Add the environment variables above.
6. For persistent subscribers and duplicate history across redeploys, attach a persistent disk and set `DATA_DIR` to its mount path.
7. Deploy and open `/`, `/health`, and `/ready` on the service URL.
8. Open your Telegram bot and send `/start`, then `/news`. Send a Telegram voice note or audio attachment to receive a text transcript.

## Important reliability notes
- The bot uses Telegram long polling, not a webhook. Do not run multiple instances/workers using the same bot token.
- The scheduled job runs only while the Render service is running. Check your Render plan's sleep/restart behavior. For robust scheduled delivery, use an external scheduler/cron service that calls a protected job endpoint (not included in this version).
- NewsAPI results are candidate discovery, not proof. The app tries to read the article page and find other coverage; GROQ assesses supplied evidence. The code only publishes stories that pass the conservative gates. This can result in zero stories if evidence is insufficient.
- No AI fact-checking pipeline can guarantee 100% accuracy. Human review is recommended for high-impact claims.
- NewsAPI and Groq availability, quotas, pricing, and model access depend on your accounts/plans.
- Speech-to-text uses Hugging Face Inference Providers and Whisper. Provider access, quota, and pricing may vary; audio downloads through the Telegram Bot API are limited to 20 MB in this implementation. Transcription errors should be handled by retrying with a shorter/clearer recording.
- Never commit API keys or Telegram tokens to GitHub. Add them only as Render environment variables.
