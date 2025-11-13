import base64
from pathlib import Path

from openai import OpenAI, OpenAIError
from openai.types import Embedding
from openai._types import NOT_GIVEN
from openai.types.chat import ChatCompletion
from typing import Union, List, Optional, Dict


class OPENAIAGENT:
    def __init__(self, openai_api_key, system_message=None, model=None):
        print(f"Initializing OPENAIAGENT chat or embed for model {model}.")

        self.model = model
        # self.system_message = system_message or "You are a helpful assistant"
        # self.history = [{"role": "system", "content": self.system_message, }]
        self.history = []
        self.last_token_usage = None

        try:
            self.client = OpenAI(api_key=openai_api_key)

            try:
                _ = self.client.models.list()
                print(f"OPENAIAGENT initialized successfully for model {model}.")
            except Exception as e:
                print("Failed to verify OpenAI API key", e)
                raise ValueError(f"Invalid or unauthorized API key: {e}") from e

        except OpenAIError as e:
            print("Failed to initialise OpenAI client", e)
            raise ValueError(f"Failed to initialise OpenAI client: {e}") from e

        except Exception as e:
            print("Unexpected error during OPENAIAGENT initialization", e)
            raise ValueError(f"Unexpected error while initializing client: {e}") from e

    def ask(self, user_input, response_type: str = None):
        user_message = None
        print("Entering OPENAIAGENT.ask()")

        if not user_input:
            print("Empty user input provided to OPENAIAGENT.ask()")
            raise ValueError("ask agent failed Please enter a valid user_input.")

        # 🔹 Handle text + image multimodal input
        if isinstance(user_input, dict) and "prompt" in user_input:
            # expected format: {"prompt": "...", "images": ["img1.png", "img2.jpg"]}
            content = [{"type": "text", "text": user_input["prompt"]}]
            for img_path in user_input.get("images", []):
                path = Path(img_path)
                if not path.exists():
                    raise FileNotFoundError(f"Image not found: {img_path}")
                mime = "image/jpeg" if path.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"}
                })
                user_message = {"role": "user", "content": content}

        else:
            user_message = {"role": "user", "content": user_input}

        self.history.append(user_message)

        try:
            if response_type in {"json_object", "json_schema"}:
                response_format = {"type": response_type}
            else:
                response_format = NOT_GIVEN

            response: ChatCompletion = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                response_format=response_format
            )

            print("Model response received successfully.")
            message = response.choices[0].message
            reply = (message.content or "").strip()

            print(f"OPENAIAGENT.ask() completed successfully with reply length {len(reply)} chars.")
            return reply
        except OpenAIError as e:
            if "context_length_exceeded" in str(e):
                print("Context length exceeded. Clearing history and retrying...")
                self.clear_conversation_history()
                self.history.append({"role": "user", "content": user_input})
                return self._ask_no_stream(user_input=user_input)
            else:
                print("OpenAI chat completion failed", e)
                raise ValueError(f"OpenAI chat completion failed: {e}") from e

        except Exception as e:
            print("Unexpected error during chat completion", e)
            raise ValueError(f"Unexpected error during chat completion: {e}") from e

    def clear_conversation_history(self):
        """Clear the conversation history but keep the system message."""
        print("Clearing OPENAIAGENT conversation history...")

        if not hasattr(self, "system_message") or not self.system_message.strip():
            print("Cannot clear conversation history: missing or empty system message.")
            raise ValueError("Cannot clear conversation history — system message is missing or empty.")

        old_length = len(self.history)
        print(f"Previous conversation length: {old_length} message(s).")

        # Reset only user/assistant messages — keep the system role for context
        self.history = [{"role": "system", "content": self.system_message}]

        print("Conversation history cleared successfully.")

    def get_token_usage(self):
        """
        Return token usage from the last API call.
        Returns (prompt_tokens, completion_tokens, total_tokens) or None if unavailable.
        """
        print("Entering OPENAIAGENT.get_token_usage()")

        if self.last_token_usage:
            prompt = getattr(self.last_token_usage, "prompt_tokens", 0)
            completion = getattr(self.last_token_usage, "completion_tokens", 0)
            total = getattr(self.last_token_usage, "total_tokens", 0)

            print(
                f"Token usage retrieved: prompt={prompt}, completion={completion}, total={total}"
            )
            return prompt, completion, total

        print("No token usage data available in OPENAIAGENT.")
        return None

class GROKAGENT:
    pass

class GEMINIAGENT:
    pass

class CLAUDEAGENT:
    pass
