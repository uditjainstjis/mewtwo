import os
import subprocess
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Load the environment variables mapping your bot token
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"🚀 *Mewtwo Autonomous Lab Online* 🚀\nWelcome {user.mention_html()}! Your GPU rig is linked.\n\nCommands:\n/status - RTX 5090 Telemetry\n/logs - Subspace Training Stream\n/idea [text] - Pipe hypothesis to engine",
    )

async def statusCommand(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends the output of nvidia-smi"""
    try:
        # Run nvidia-smi securely and read stdout
        output = subprocess.check_output(
            ["nvidia-smi"], 
            text=True, 
            timeout=5
        )
        # Discord/Telegram have message length limits, but nvidia-smi easily fits.
        await update.message.reply_text(f"```text\n{output}\n```", parse_mode='MarkdownV2')
    except Exception as e:
        await update.message.reply_text(f"⚠️ Error fetching GPU status: {e}")

async def logsCommand(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends the last 20 lines of the ML Training Logs"""
    try:
        path = os.path.join(os.path.dirname(__file__), "train_matrix.log")
        output = subprocess.check_output(
            ["tail", "-n", "20", path], 
            text=True, 
            timeout=5
        )
        await update.message.reply_text(f"**CF-LoRA Training Matrix Stream:**\n```text\n{output}\n```", parse_mode='MarkdownV2')
    except Exception as e:
        await update.message.reply_text(f"⚠️ Error fetching logs: {e}")

async def ideaCommand(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Saves a user idea from the phone to a local backlog for the researcher"""
    idea_text = " ".join(context.args)
    if not idea_text:
        await update.message.reply_text("Please provide an idea: /idea Try applying rank 128 to medical.")
        return
        
    path = os.path.join(os.path.dirname(__file__), "hypotheses_backlog.txt")
    with open(path, "a") as f:
        f.write(f"- {idea_text}\n")
    
    await update.message.reply_text("✅ Idea received and queued into the Research Engine Backlog.")

def main() -> None:
    """Start the bot."""
    if not TOKEN:
        print("TELEGRAM_BOT_TOKEN not found in .env file! Exiting...")
        return

    print("🐕 Mewtwo Remote Bridge is Initializing...")
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("status", statusCommand))
    application.add_handler(CommandHandler("logs", logsCommand))
    application.add_handler(CommandHandler("idea", ideaCommand))

    # Run the bot until the user presses Ctrl-C
    print("✅ Webhook active. Awaiting your messages...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
