import google.generativeai as genai
import logging
import os

class GeminiHelper:
    """
    Gemini helper class.
    """

    def __init__(self, config: dict):
        """
        Initializes the Gemini helper class with the given configuration.
        :param config: A dictionary containing the Gemini configuration
        """
        # Configure the Gemini API key
        api_key = config.get('api_key')
        if not api_key:
            raise ValueError("Gemini API key is missing from the configuration.")
        genai.configure(api_key=api_key)

        # The user wants to use Gemini 1.5 Pro, but the model name in the API might be different.
        # Let's use 'gemini-1.5-pro-latest' for now.
        self.model = genai.GenerativeModel('gemini-1.5-pro-latest')
        self.config = config
        self.conversations: dict[int, list] = {}  # {chat_id: history}

    async def get_chat_response(self, chat_id: int, query: str) -> tuple[str, str]:
        """
        Gets a full response from the Gemini model.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :return: The answer from the model and the number of tokens used (dummy value for now)
        """
        logging.info(f"Getting chat response for chat_id: {chat_id} with query: {query}")

        if chat_id not in self.conversations:
            self.conversations[chat_id] = []

        try:
            # Start a chat session with the history
            chat = self.model.start_chat(history=self.conversations.get(chat_id, []))

            # Send the new message
            response = await chat.send_message_async(query)
            answer = response.text

            # Update the conversation history
            self.conversations[chat_id] = chat.history

            # Dummy token count for now. Gemini API has a different way of counting tokens.
            token_count = await self.model.count_tokens_async(chat.history)

            return answer, str(token_count.total_tokens)

        except Exception as e:
            logging.error(f"Error getting response from Gemini: {e}")
            raise Exception(f"⚠️ An error occurred while communicating with Gemini. ⚠️\n{str(e)}") from e

    def reset_chat_history(self, chat_id):
        """
        Resets the conversation history for a given chat_id.
        """
        if chat_id in self.conversations:
            del self.conversations[chat_id]
        logging.info(f"Chat history for chat_id {chat_id} has been reset.")

    async def get_conversation_stats(self, chat_id: int) -> tuple[int, int]:
        """
        Gets the number of messages and tokens used in the conversation.
        :param chat_id: The chat ID
        :return: A tuple containing the number of messages and tokens used
        """
        if chat_id not in self.conversations:
            self.conversations[chat_id] = []

        token_count = await self.model.count_tokens_async(self.conversations[chat_id])
        return len(self.conversations[chat_id]), token_count.total_tokens

    async def get_chat_response_stream(self, chat_id: int, query: str):
        """
        Stream response from the Gemini model.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :return: The answer from the model and the number of tokens used, or 'not_finished'
        """
        logging.info(f"Getting chat response for chat_id: {chat_id} with query: {query}")

        if chat_id not in self.conversations:
            self.conversations[chat_id] = []

        try:
            # Start a chat session with the history
            chat = self.model.start_chat(history=self.conversations.get(chat_id, []))

            # Send the new message and stream the response
            response_stream = await chat.send_message_async(query, stream=True)

            answer = ''
            async for chunk in response_stream:
                if chunk.text:
                    answer += chunk.text
                    yield answer, 'not_finished'

            # Update the conversation history
            self.conversations[chat_id] = chat.history

            token_count = await self.model.count_tokens_async(chat.history)
            yield answer, str(token_count.total_tokens)

        except Exception as e:
            logging.error(f"Error getting response from Gemini: {e}")
            raise Exception(f"⚠️ An error occurred while communicating with Gemini. ⚠️\n{str(e)}") from e
