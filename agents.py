import base64
from pathlib import Path
import httpx
from openai import OpenAI, OpenAIError
from openai._types import NOT_GIVEN
from openai.types.chat import ChatCompletion
from typing import Union, List, Optional, Dict

from google import genai
from google.genai import types
from google.genai.errors import APIError


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
                return self.ask(user_input=user_input)
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


class GROKAGENT():

    def __init__(self, grok_api_key, system_message=None, model=None):
        print(f"Initializing GROKAGENT chat model {model}.")

        self.model = model
        # self.system_message = system_message or "You are a helpful assistant"
        # self.history = [{"role": "system", "content": self.system_message, }]
        self.history = []
        self.last_token_usage = None

        try:
            self.client = OpenAI(
                api_key=grok_api_key,
                base_url="https://api.x.ai/v1",
                timeout=httpx.Timeout(3600.0)
            )

            try:
                _ = self.client.models.list()
                print(f"GROKAGENT  initialized successfully for model {model}.")
            except Exception as e:
                print("Failed to verify GROK xAI API key", e)
                raise ValueError(f"Invalid or unauthorized API key: {e}") from e

        except OpenAIError as e:
            print("Failed to initialise GROK xAI API client", e)
            raise ValueError(f"Failed to initialise GROK xAI API client: {e}") from e

        except Exception as e:
            print("Unexpected error during GROKAGENT initialization", e)
            raise ValueError(f"Unexpected error while initializing client: {e}") from e

    def ask(self, user_input, response_type: str = None):
        user_message = None
        print("Entering GROKAGENT.ask()")

        if not user_input:
            print("Empty user input provided to GROKAGENT.ask()")
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

            print(f"GROKAGENT.ask() completed successfully with reply length {len(reply)} chars.")
            return reply
        except OpenAIError as e:
            if "context_length_exceeded" in str(e):
                print("Context length exceeded. Clearing history and retrying...")
                self.clear_conversation_history()
                self.history.append({"role": "user", "content": user_input})
                return self.ask(user_input=user_input)
            else:
                print("GROK xAI chat completion failed", e)
                raise ValueError(f"GROK xAI chat completion failed: {e}") from e

        except Exception as e:
            print("Unexpected error during chat completion", e)
            raise ValueError(f"Unexpected error during chat completion: {e}") from e

    def clear_conversation_history(self):
        """Clear the conversation history but keep the system message."""
        print("Clearing GROKAGENT conversation history...")

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
        print("Entering GROKAGENT.get_token_usage()")

        if self.last_token_usage:
            prompt = getattr(self.last_token_usage, "prompt_tokens", 0)
            completion = getattr(self.last_token_usage, "completion_tokens", 0)
            total = getattr(self.last_token_usage, "total_tokens", 0)

            print(
                f"Token usage retrieved: prompt={prompt}, completion={completion}, total={total}"
            )
            return prompt, completion, total

        print("No token usage data available in GROKAGENT.")
        return None


class DEEPSEEKAGENT():

    def __init__(self, deep_seek_api_key, system_message=None, model=None):
        print(f"Initializing DEEPSEEKAGENT chat model {model}.")

        self.model = model
        # self.system_message = system_message or "You are a helpful assistant"
        # self.history = [{"role": "system", "content": self.system_message, }]
        self.history = []
        self.last_token_usage = None

        try:
            self.client = OpenAI(
                api_key=deep_seek_api_key,
                base_url="https://api.deepseek.com/v1"
            )

            try:
                _ = self.client.models.list()
                print(f"DEEPSEEKAGENT  initialized successfully for model {model}.")
            except Exception as e:
                print("Failed to verify DEEPSEEKAGENT API key", e)
                raise ValueError(f"Invalid or unauthorized API key: {e}") from e

        except OpenAIError as e:
            print("Failed to initialise DEEPSEEKAGENT API client", e)
            raise ValueError(f"Failed to initialise DEEPSEEKAGENT API client: {e}") from e

        except Exception as e:
            print("Unexpected error during DEEPSEEKAGENT initialization", e)
            raise ValueError(f"Unexpected error while initializing client: {e}") from e

    def ask(self, user_input, response_type: str = None):
        user_message = None
        print("Entering DEEPSEEKAGENT.ask()")

        if not user_input:
            print("Empty user input provided to DEEPSEEKAGENT.ask()")
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

            print(f"DEEPSEEKAGENT.ask() completed successfully with reply length {len(reply)} chars.")
            return reply
        except OpenAIError as e:
            if "context_length_exceeded" in str(e):
                print("Context length exceeded. Clearing history and retrying...")
                self.clear_conversation_history()
                self.history.append({"role": "user", "content": user_input})
                return self.ask(user_input=user_input)
            else:
                print("DEEPSEEKAGENT chat completion failed", e)
                raise ValueError(f"DEEPSEEKAGENT chat completion failed: {e}") from e

        except Exception as e:
            print("Unexpected error during chat completion", e)
            raise ValueError(f"Unexpected error during chat completion: {e}") from e

    def clear_conversation_history(self):
        """Clear the conversation history but keep the system message."""
        print("Clearing DEEPSEEKAGENT conversation history...")

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
        print("Entering DEEPSEEKAGENT.get_token_usage()")

        if self.last_token_usage:
            prompt = getattr(self.last_token_usage, "prompt_tokens", 0)
            completion = getattr(self.last_token_usage, "completion_tokens", 0)
            total = getattr(self.last_token_usage, "total_tokens", 0)

            print(
                f"Token usage retrieved: prompt={prompt}, completion={completion}, total={total}"
            )
            return prompt, completion, total

        print("No token usage data available in DEEPSEEKAGENT.")
        return None


class QWENAGENT:

    def __init__(self, deep_seek_api_key, system_message=None, model=None):
        print(f"Initializing QWENAGENT chat model {model}.")

        self.model = model
        # self.system_message = system_message or "You are a helpful assistant"
        # self.history = [{"role": "system", "content": self.system_message, }]
        self.history = []
        self.last_token_usage = None

        try:
            self.client = OpenAI(
                api_key=deep_seek_api_key,
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            )

            try:
                _ = self.client.models.list()
                print(f"QWENAGENT  initialized successfully for model {model}.")
            except Exception as e:
                print("Failed to verify QWENAGENT API key", e)
                raise ValueError(f"Invalid or unauthorized API key: {e}") from e

        except OpenAIError as e:
            print("Failed to initialise QWENAGENT API client", e)
            raise ValueError(f"Failed to initialise QWENAGENT API client: {e}") from e

        except Exception as e:
            print("Unexpected error during QWENAGENT initialization", e)
            raise ValueError(f"Unexpected error while initializing client: {e}") from e

    def ask(self, user_input, response_type: str = None):
        user_message = None
        print("Entering QWENAGENT.ask()")

        if not user_input:
            print("Empty user input provided to QWENAGENT.ask()")
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

            print(f"QWENAGENT.ask() completed successfully with reply length {len(reply)} chars.")
            return reply
        except OpenAIError as e:
            if "context_length_exceeded" in str(e):
                print("Context length exceeded. Clearing history and retrying...")
                self.clear_conversation_history()
                self.history.append({"role": "user", "content": user_input})
                return self.ask(user_input=user_input)
            else:
                print("QWENAGENT chat completion failed", e)
                raise ValueError(f"QWENAGENT chat completion failed: {e}") from e

        except Exception as e:
            print("Unexpected error during chat completion", e)
            raise ValueError(f"Unexpected error during chat completion: {e}") from e

    def clear_conversation_history(self):
        """Clear the conversation history but keep the system message."""
        print("Clearing QWENAGENT conversation history...")

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
        print("Entering QWENAGENT.get_token_usage()")

        if self.last_token_usage:
            prompt = getattr(self.last_token_usage, "prompt_tokens", 0)
            completion = getattr(self.last_token_usage, "completion_tokens", 0)
            total = getattr(self.last_token_usage, "total_tokens", 0)

            print(
                f"Token usage retrieved: prompt={prompt}, completion={completion}, total={total}"
            )
            return prompt, completion, total

        print("No token usage data available in QWENAGENT.")
        return None


class GEMINIAGENT:

    def __init__(self, gemeni_api_key: str, system_message: str = None, model: str = None):
        """Initializes the Gemini client and chat session."""
        print(f"Initializing GEMINIAGENT chat for model {model}.")

        self.model = model
        self.system_message = system_message
        self.history = []  # types.Content objects will be managed by the chat session.
        self.last_usage_metadata = None  # For token usage retrieval
        self.chat_session = None

        try:
            # 1. Initialization: Client setup uses the API key
            self.client = genai.Client(api_key=gemeni_api_key)

            # Optional: Verify API key by listing models (similar to OpenAI logic)
            # The genai client does not have a simple 'models.list()'. A simple call
            # with a quick timeout can serve as a proxy for key validation.
            # For simplicity, we skip this complex step but rely on the subsequent
            # chat creation to validate the key and connection.

            # 2. Chat Session Creation: This is where conversation history is managed
            # The system instruction is part of the initial config.
            config = None
            if self.system_message:
                config = types.GenerateContentConfig(
                    system_instruction=self.system_message
                )

            self.chat_session = self.client.chats.create(
                model=self.model,
                config=config,
            )

            print(f"GEMINIAGENT initialized successfully for model {model}.")

        except APIError as e:
            print("Failed to initialize Gemini client", e)
            raise ValueError(f"Invalid or unauthorized API key or connection error: {e}") from e
        except Exception as e:
            print("Unexpected error during GEMINIAGENT initialization", e)
            raise ValueError(f"Unexpected error while initializing client: {e}") from e

    def ask(self, user_input: Union[str, Dict[str, Union[str, List[str]]]], response_type: str = None) -> str:
        """Sends a message to the model and returns the text response."""
        print("Entering GEMINIAGENT.ask()")

        if not user_input:
            print("Empty user input provided to GEMINIAGENT.ask()")
            raise ValueError("ask agent failed: Please enter a valid user_input.")

        # 1. Build the 'contents' list (list of Parts)
        parts: List[types.Part] = []

        # 🔹 Handle text + image multimodal input
        if isinstance(user_input, dict) and "prompt" in user_input:
            # expected format: {"prompt": "...", "images": ["img1.png", "img2.jpg"]}

            # Add text part
            parts.append(types.Part.from_text(user_input["prompt"]))

            # Add image parts (using local file paths)
            for img_path in user_input.get("images", []):
                path = Path(img_path)
                if not path.exists():
                    raise FileNotFoundError(f"Image not found: {img_path}")

                # Determine MIME type
                mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
                mime = mime_map.get(path.suffix.lower(), "application/octet-stream")

                with open(path, "rb") as f:
                    image_bytes = f.read()

                # Use Part.from_bytes for local image files
                parts.append(types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime
                ))

        else:
            # Handle text-only input
            parts.append(types.Part.from_text(user_input))

        # 2. Configure the generation request
        config = types.GenerateContentConfig()

        # 3. Handle JSON/Structured Output request
        if response_type == "json_object":
            # Gemini uses response_mime_type for JSON output
            # A schema can also be provided, but for simple JSON_OBJECT, this is the minimal change
            config.response_mime_type = "application/json"

        # 'json_schema' is a more complex case involving response_schema and is omitted for brevity,
        # but would map to types.GenerateContentConfig(response_schema=...)

        # 4. Send the message
        try:
            # Use the chat session's send_message method to automatically manage history
            response = self.chat_session.send_message(
                contents=parts,
                config=config if any(getattr(config, f, None) for f in
                                     ['response_mime_type', 'response_schema', 'thinking_config']) else None
                # Only pass config if configured
            )

            print("Model response received successfully.")
            reply = response.text.strip()

            # Save token usage metadata
            self.last_usage_metadata = response.usage_metadata

            print(f"GEMINIAGENT.ask() completed successfully with reply length {len(reply)} chars.")
            return reply

        except APIError as e:
            # Handle context length exceeded specifically
            # Gemini's APIError will contain context limits, but a universal retry is more complex
            # than the OpenAI one. The most direct equivalent is to clear and re-raise.
            if "context_length" in str(e).lower() or "quota" in str(e).lower():
                print("Context length or quota issue detected. Clearing history and retrying with current input...")
                self.clear_conversation_history()
                # Re-send the *current* user input, which includes the multimodal handling logic above.
                return self.ask(user_input=user_input, response_type=response_type)
            else:
                print("Gemini chat completion failed", e)
                raise ValueError(f"Gemini chat completion failed: {e}") from e

        except Exception as e:
            print("Unexpected error during chat completion", e)
            raise ValueError(f"Unexpected error during chat completion: {e}") from e

    def clear_conversation_history(self):
        """Clear the conversation history by recreating the chat session."""
        print("Clearing GEMINIAGENT conversation history...")

        if self.chat_session is None:
            # Recreate a new client and chat session if somehow lost
            self.__init__(self.client._client._base_url, self.system_message, self.model)
            return

        old_length = len(self.chat_session.get_history().history)
        print(f"Previous conversation length: {old_length} message(s).")

        # Reset by creating a new chat session with the original model and config
        config = types.GenerateContentConfig(system_instruction=self.system_message) if self.system_message else None
        self.chat_session = self.client.chats.create(
            model=self.model,
            config=config,
        )

        print("Conversation history cleared successfully.")

    def get_token_usage(self):
        """
        Return token usage from the last API call using usage_metadata.
        Returns (prompt_tokens, completion_tokens, total_tokens) or None if unavailable.
        """
        print("Entering GEMINIAGENT.get_token_usage()")

        if self.last_usage_metadata:
            # The GenAI SDK returns tokens under 'usage_metadata'
            prompt = self.last_usage_metadata.prompt_token_count or 0
            completion = self.last_usage_metadata.candidates_token_count or 0
            total = self.last_usage_metadata.total_token_count or 0

            print(
                f"Token usage retrieved: prompt={prompt}, completion={completion}, total={total}"
            )
            return prompt, completion, total

        print("No token usage data available in GEMINIAGENT.")
        return None
