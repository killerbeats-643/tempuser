import os
import json
import base64
import logging
import google.generativeai as genai
import fasttext
from typing import Optional, List, Any

from dotenv import load_dotenv
load_dotenv()

# Constants
FASTTEXT_MODEL_PATH = "lid.176.bin"

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ChatService")

# Load Language Detection Model once at module load time
try:
    lang_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
    logger.info("FastText language detection model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load FastText model: {e}", exc_info=True)
    lang_model = None


class ChatService:
    """
    ChatService encapsulates interaction with Google's Gemini Generative AI model,
    and provides language detection and prompt preparation utilities.
    """


    @staticmethod
    def configure_api() -> Optional[str]:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY environment variable not set.")
            return None
        genai.configure(api_key=api_key)
        logger.info("Gemini API key configured successfully.")
        return api_key


    @classmethod
    def load_model(cls, model_name: Optional[str] = None) -> None:
        """
        Loads the generative model specified by 'model_name' argument or
        falls back to environment variable 'MODEL'.
        Raises an exception if model loading fails.
        """
        if not model_name:
            model_name = os.getenv("MODEL")
        if not model_name:
            error_msg = "MODEL environment variable not set and no model_name provided."
            logger.error(error_msg)
            raise EnvironmentError(error_msg)

        try:
            cls.model = genai.GenerativeModel(model_name)
            logger.info(f"Model '{model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}", exc_info=True)
            raise

    @staticmethod
    def detect_language(text: str) -> str:
        """
        Detects the language of the given text using FastText language detection model.
        Defaults to 'en' if detection fails or model is not loaded.
        """
        if not lang_model:
            logger.warning("FastText model not loaded. Defaulting language to 'en'.")
            return "en"
        #fallback use langdetect
        try:
            predictions = lang_model.predict(text, k=1)
            lang_code = predictions[0][0].replace("__label__", "")
            logger.info(f"Detected language: {lang_code}")
            return lang_code
        except Exception as e:
            logger.warning(f"Language detection failed: {e}. Defaulting to 'en'.")
            return "en"

    @staticmethod
    def _pdf_to_base64(pdf_bytes: bytes) -> str:
        """
        Converts raw PDF bytes to base64 encoded ASCII string.
        """
        if not pdf_bytes:
            return ""
        return base64.b64encode(pdf_bytes).decode("ascii")

    @staticmethod
    def _prepare_prompt(message: str, chat_history: Optional[List[Any]], data: Optional[str]) -> str:
        """
        Prepares the prompt string to send to the Gemini model.
        Incorporates chat history, detected language, and raw inspection data.
        """
        lang_code = ChatService.detect_language(message)
        history_text = ""

        if chat_history:
            entries = []
            for msg in chat_history:
                # Support flexible message format: check for isUser and content attributes
                role = "User" if getattr(msg, "isUser", False) else "Assistant"
                content = getattr(msg, "content", "")
                # If content is a dict, convert to JSON string to avoid raw dict dumps
                if isinstance(content, dict):
                    content = json.dumps(content)
                entries.append(f"{role}: {content}")
            history_text = "\n".join(entries)

        prompt = ""
        if history_text:
            prompt += (
                "You are SEYO Bot, a multilingual business assistant, your main functionalities are to help the users "
                "with the business knowledge from the document attached and the message history. Below is the previous "
                "conversation for your context:\n"
                f"{history_text}\n\n"
                f"The detected user input language is '{lang_code}', the response must be in the user input language or detected language. "
                f"The raw inspection data is given as: {data}.\n\n"
            )

        prompt += (
            f"User's current input:\n{message}\n\n"
            "Please follow this approach:\n"
            "1. Analyze the user's question carefully and determine their intent.\n"
            "2. If the user is asking a general business-related or casual question (related to inspection processes, business operations, or workflows), respond with clear, professional, and friendly language using general business knowledge from the document or chat history.\n"
            "3. If the user's request is about extracting or analyzing information from a document or inspection data (e.g., asking about values, numbers, observations, or inspection reports), then strictly refer to the uploaded PDF file and the chat history to answer accurately.\n"
            "4. Do NOT answer questions outside the scope of the document, business, or inspection content. But strictly work like a chatbot. If user is greeting and casually asking questions, respond casually. Don't always say the entire description. Be a business chatbot.\n"
            "5. Maintain a friendly, multilingual tone suitable for business users. Prioritize clarity and professionalism.\n"
            "6. If the requested information is not found in the document or chat history, politely state that it is not mentioned or available.\n\n"
            "Now, provide your response based on the user's intent and the available context and detected language."
        )

        return prompt

    @staticmethod
    def _prepare_payload(prompt: str, pdf_base64: str) -> List[dict]:
        """
        Constructs the content payload for the Gemini model request.
        """
        parts = [{"text": prompt}]
        if pdf_base64:
            parts.append({
                "inline_data": {
                    "mime_type": "application/pdf",
                    "data": pdf_base64
                }
            })
        return [{
            "role": "user",
            "parts": parts
        }]

    @classmethod
    def get_chatresponse(
        cls,
        pdf_bytes: Optional[bytes],
        message: str,
        chat_history: Optional[List[Any]] ,
        data: Optional[str]
    ) -> str:
        """
        Get chat response from Gemini model based on user message and optional PDF + history.
        """
        
        ChatService.load_model()
        ChatService.configure_api()
        
   
        try:
            logger.info(f"Received message: {message}")
            prompt = cls._prepare_prompt(message, chat_history, data)
            pdf_base64 = cls._pdf_to_base64(pdf_bytes) if pdf_bytes else ""
            content = cls._prepare_payload(prompt, pdf_base64)

            generation_config = {
                "temperature": 0.2,
                "top_p": 0.8,
                "max_output_tokens": 2048
            }

            logger.info("Sending request to Gemini...")
            response = cls.model.generate_content(
                content,
                generation_config=generation_config,
                safety_settings=[]
            )

            if hasattr(response, "text"):
                logger.info(f"Response received (first 50 chars): {response.text[:50]}")
                return response.text
            else:
                logger.error("Unexpected response format from Gemini.")
                return "Error: Unexpected response format from Gemini API."

        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}", exc_info=True)
            return f"Error: Unable to generate response due to: {str(e)}"
