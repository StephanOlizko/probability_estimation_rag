

class PromptFactory:
    """
    A factory class to create various types of prompts.
    """

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
                    "Respond strictly in the following JSON format:\n"
                    "{\n"
                    '  "probability": <number>,\n'
                    '  "justification": "<short explanation>"\n'
                    "}\n"
                    "Do not include any other text."
                )
            },
            {
                "role": "user",
                "content": f"Event: {query}\nWhat is the probability this event will occur? Please follow the required JSON format."
            }
        ]