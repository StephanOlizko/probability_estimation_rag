from typing import List, Optional
import json
from pydantic import BaseModel


class ResponseProbJustification(BaseModel):
    probability: float
    justification: str


class PromptFactory:
    """
    A factory class to create various types of prompts.
    """

    @staticmethod
    def create_general_topic_prompt(query: str) -> list:
        """
        Creates a prompt to get a general topic for the given text.

        :param query: The text to get a general topic for.
        :return: List of messages to pass to LLM.
        """
        return [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides a general topic for the text. Your output should be a few words with no other text."
            },
            {
                "role": "user",
                "content": f"Please provide a general topic for the following text in a few words: \n\n {query}"
            }
        ]
        


    @staticmethod
    def create_probability_prompt(query: str) -> list:
        """
        Формирует список сообщений для LLM, чтобы получить вероятность бинарного события и обоснование в формате JSON.

        :param query: Вопрос о бинарном событии.
        :return: Список сообщений для передачи в LLM.
        """
        return [
            {
                "role": "system",
                "content": (
                    "You are an expert forecaster. "
                    "Given a binary event, estimate the probability (0-100%) that it will happen, "
                    "and provide a brief justification for your estimate. "
                    "Respond strictly in the JSON format."
                    f"The JSON object must use the schema: : {json.dumps(ResponseProbJustification.model_json_schema(), indent=2)}"
                    
                )
            },
            {
                "role": "user",
                "content": f"Event: {query}\n\n What is the probability this event will occur? Please follow the required JSON format."
            }
        ]

    @staticmethod
    def create_probability_prompt_with_context(query, context):
        """
        Формирует список сообщений для LLM, чтобы получить вероятность бинарного события и обоснование в формате JSON.

        :param query: Вопрос о бинарном событии.
        :param context_list: Контекст для LLM.
        :return: Список сообщений для передачи в LLM.
        """
        return [
            {
                "role": "system",
                "content": (
                    "You are an expert forecaster. "
                    "Given a binary event, estimate the probability (0-100%) that it will happen, "
                    "and provide a brief justification for your estimate. "
                    "Respond strictly in the JSON format."
                    f"The JSON object must use the schema: : {json.dumps(ResponseProbJustification.model_json_schema(), indent=2)}"
                    
                )
            },
            {
                "role": "user",
                "content": f"Event: {query}\n\n Context: {context}\n\n What is the probability this event will occur? Please follow the required JSON format."
            }
        ]