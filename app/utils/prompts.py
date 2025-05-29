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

    @staticmethod
    def create_pro_arguments_prompt_with_context(query: str, context: str) -> list:
        return [
            {
                "role": "system",
                "content": (
                    "You are an expert analyst. Your task is to argue **why** the given binary event is **likely** to happen, "
                    "based on the provided context. Focus only on supportive evidence and logical reasoning. "
                    "Don't mention opposing views."
                )
            },
            {
                "role": "user",
                "content": f"Event: {query}\n\nContext:\n{context}\n\nWhy is this event **likely** to happen?"
            }
        ]

    @staticmethod
    def create_con_arguments_prompt_with_context(query: str, context: str) -> list:
        return [
            {
                "role": "system",
                "content": (
                    "You are an expert analyst. Your task is to argue **why** the given binary event is **unlikely** to happen, "
                    "based on the provided context. Focus only on counter-evidence and skeptical reasoning. "
                    "Don't mention supportive views."
                )
            },
            {
                "role": "user",
                "content": f"Event: {query}\n\nContext:\n{context}\n\nWhy is this event **unlikely** to happen?"
            }
        ]

    @staticmethod
    def create_final_reasoning_prompt_with_arguments(query: str, pro_argument: str, con_argument: str) -> list:
        return [
            {
                "role": "system",
                "content": (
                    "You are a forecasting assistant combining multiple lines of reasoning. "
                    "Given arguments both for and against a binary event, assess the overall probability (0-100%) of the event happening. "
                    "Respond strictly in JSON format using this schema:\n"
                    f"{json.dumps(ResponseProbJustification.model_json_schema(), indent=2)}"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Event: {query}\n\n"
                    f"Arguments supporting the event:\n{pro_argument}\n\n"
                    f"Arguments against the event:\n{con_argument}\n\n"
                    "What is the most likely outcome? Provide final reasoning and a probability estimate."
                )
            }
        ]