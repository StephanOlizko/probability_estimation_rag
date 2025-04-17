from utils.helpers import OpenRouterClient
from utils.prompts import PromptFactory
from logger import setup_logger
import logging
import json

logger = logging.getLogger("retrieval_pipeline")

class NoRagLLM:
    """
    A class that simulates a non-RAG LLM by returning the input text as the output.
    """

    def __init__(self, model_name=None):
        """
        Initializes the NoRagLLM class.
        """
        self.model_name = model_name
        self.client = OpenRouterClient(model=model_name)
        self.logger = setup_logger()
        self.logger.info(f"NoRagLLM initialized with model: {model_name}")

    def generate_anwser(self, query):
        self.logger.info(f"Generating answer for query: {query}")
        messages = PromptFactory.create_probability_prompt(query)
        response_format = {"type": "json_object"}
        self.logger.info(f"Messages for LLM: {messages}")
        try:
            response = self.client.generate_response(messages=messages, response_format=response_format)
            self.logger.info(f"Response received: {response}")

            content = response["choices"][0]["message"]["content"]
            self.logger.info(f"Raw content: {content}")
            
            # Extract JSON from the response if it contains additional text
            try:
                # First try direct JSON parsing
                response_json = json.loads(content)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON portion using regex
                import re
                json_match = re.search(r'(\{.*?\})', content, re.DOTALL)
                if json_match:
                    try:
                        response_json = json.loads(json_match.group(1))
                        self.logger.info(f"Extracted JSON from text: {response_json}")
                    except json.JSONDecodeError:
                        self.logger.error("Failed to parse extracted JSON")
                        return "Unknown", f"Could not parse model response as JSON"
                else:
                    self.logger.error("No JSON found in response")
                    return "Unknown", "Model did not return JSON format"
            
            self.logger.info(f"Parsed response: {response_json}")
            return {"probability": response_json["probability"], "justification": response_json["justification"]}

        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            raise e

if __name__ == "__main__":
    model_name = "deepseek/deepseek-r1:free"

    query = "Will the stock market crash in 2024?"
    no_rag_llm = NoRagLLM(model_name=model_name)
    print(no_rag_llm.generate_anwser(query))