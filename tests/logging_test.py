import logging
from logging.handlers import TimedRotatingFileHandler
import sys
from pathlib import Path

Path("logs").mkdir(exist_ok=True)


def setup_daily_logger():
    logger = logging.getLogger("DailyTerminalLogger")
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate logs if the handler is already initialized
    if logger.handlers:
        return logger

    # 1. Define how your logs will look (Time - Type of message - The actual message)
    log_format = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # 2. Set up the daily rotating file handler
    # 'when="midnight"' creates a new file every night.
    # 'interval=1' means it happens every 1 midnight.
    # 'backupCount=30' keeps up to 30 days of logs automatically before deleting old ones.
    file_handler = TimedRotatingFileHandler(
        filename="logs/terminal_log.log", 
        when="midnight", 
        interval=1, 
        backupCount=30,
        encoding="utf-8"
    )
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    # 3. Add a second handler so you STILL see the outputs in the terminal in real-time
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_format)
    logger.addHandler(stream_handler)

    return logger

# --- How to use it in your code ---
if __name__ == "__main__":
    logger = setup_daily_logger()
    
    logger.info("Script has started successfully.")
    logger.warning("This is a warning message — something might be off.")
    logger.error("An error occurred here!")
    
    # Example of logging variables
    username = "Alice"
    logger.info(f"User {username} logged in.")