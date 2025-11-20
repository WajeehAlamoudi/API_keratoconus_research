import config
from agents import (
    OPENAIAGENT,
    GROKAGENT,
    DEEPSEEKAGENT,
    QWENAGENT,
    GEMINIAGENT
)
from ask_utils import run_agent_ask
import threading

MODEL_CONFIG = {
    "gpt5_results.json": {
        "class": OPENAIAGENT,
        "kwargs": {"api_key": config.OPENAI_API_KEY, "model": "gpt-5"}
    },
    "grok_results.json": {
        "class": GROKAGENT,
        "kwargs": {"api_key": config.GROK_API_KEY, "model": "grok-2-latest"}
    },
    "deepseek_results.json": {
        "class": DEEPSEEKAGENT,
        "kwargs": {"api_key": config.DEEPSEEK_API_KEY, "model": "deepseek-v3.2-exp"}
    },
    "qwen_results.json": {
        "class": QWENAGENT,
        "kwargs": {"api_key": config.QWEN_API_KEY, "model": "qwen-3-vl-flash"}
    },
    "gemini_results.json": {
        "class": GEMINIAGENT,
        "kwargs": {"api_key": config.GEMINI_API_KEY, "model": "gemini-3-pro-preview"}
    },
}

IMAGE_FOLDER = r"C:\Users\wajee\PycharmProjects\API_keratoconus_research\images"


threads = []

threads.append(threading.Thread(
    target=run_agent_ask,
    args=(OPENAIAGENT, {"api_key": config.OPENAI_API_KEY, "model": "gpt-5"}, "gpt5_results.json", IMAGE_FOLDER)
))

threads.append(threading.Thread(
    target=run_agent_ask,
    args=(GROKAGENT, {"api_key": config.GROK_API_KEY, "model": "grok-2-latest"}, "grok_results.json", IMAGE_FOLDER)
))

threads.append(threading.Thread(
    target=run_agent_ask,
    args=(DEEPSEEKAGENT, {"api_key": config.DEEPSEEK_API_KEY, "model": "deepseek-v3.2-exp"}, "deepseek_results.json", IMAGE_FOLDER)
))

threads.append(threading.Thread(
    target=run_agent_ask,
    args=(QWENAGENT, {"api_key": config.QWEN_API_KEY, "model": "qwen-3-vl-flash"}, "qwen_results.json", IMAGE_FOLDER)
))

threads.append(threading.Thread(
    target=run_agent_ask,
    args=(GEMINIAGENT, {"api_key": config.GEMINI_API_KEY, "model": "gemini-3-pro-preview"}, "gemini_results.json", IMAGE_FOLDER)
))

# Start all threads
for t in threads:
    t.start()

# Wait for all to finish
for t in threads:
    t.join()


print("🎉 ALL THREADS FINISHED! ALL MODELS COMPLETED!")
