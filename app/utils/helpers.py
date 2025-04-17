import requests
import json
from os import environ
from dotenv import load_dotenv
import logging
from bs4 import BeautifulSoup
import os
import sys
import logging
import logging.handlers


logger = logging.getLogger("retrieval_pipeline")

load_dotenv()


class OpenRouterClient:
    def __init__(self, model="deepseek/deepseek-chat-v3-0324:free"):
        api_key = environ.get("API_KEY")

        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.model=model

    def generate_response(self, messages=None, response_format=None):
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": messages,
        }
        if response_format is not None: data["response_format"] = response_format

        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            return response.json()
        
        except Exception as e:
            logger.error(f"Error in OpenRouterClient: {e}")
            return None    


def get_general_topic(text, model="deepseek/deepseek-chat-v3-0324:free"):    
    # This function is used to get the general topic in a few words of a text using the OpenRouter API.
    # It takes the text and model as input and returns the topic.

    client = OpenRouterClient(model=model)
    messages = [{"role": "user", "content": f"Please provide a general topic for the following text text in a few words: \n\n {text}"},
                {"role": "system", "content": "You are a helpful assistant that provides a general topic for the text. Your output should be a few words with no other text."}]
    response = client.generate_response(messages=messages)
    if response and "choices" in response and len(response["choices"]) > 0:
        return response["choices"][0]["message"]["content"]
    else:
        return None
    

def generate_response_based_on_context(messages, model="deepseek/deepseek-chat-v3-0324:free"):
    # This function is used to get the response based on the context using the OpenRouter API.
    # It takes the messages and model as input and returns the response.

    client = OpenRouterClient(model=model)
    response = client.generate_response(messages=messages)
    return response
    
def get_relevant_news_links(query, params=None, max_results=500):
    links = []

    base_url = "https://content.guardianapis.com/search"
    api_key = environ.get("GUARDIAN_API_KEY")

    if not api_key:
        logger.error("GUARDIAN_API_KEY is not set in the environment variables.")
        return []

    str_add = []
    if params is not None:
        for key, value in params.items():
            if key == "from-date":
                str_add.append(f"{key}={value}")
            elif key == "to-date":
                str_add.append(f"{key}={value}")
            elif key == "page-size":
                str_add.append(f"{key}={value}")
            elif key == "order-by":
                str_add.append(f"{key}={value}")
    
    query_str = query.replace(" ", "%20")
    str_add.append(f"q={query_str}")
    str_add.append(f"api-key={api_key}")

    url = f"{base_url}?{'&'.join(str_add)}"
    logger.debug(f"Fetching data from Guardian API with URL: {url}")

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        total_pages = data["response"]["pages"]
        logger.debug(f"Total pages to fetch: {total_pages}")
    else:
        logger.error(f"Error fetching data from Guardian API: {response.status_code} - {response.text}")
        return []

    for num_page in range(1, total_pages + 1):
        page_url = f"{base_url}?{'&'.join(str_add)}&page={num_page}"
        logger.debug(f"Fetching page {num_page} with URL: {page_url}")
        response = requests.get(page_url)
        if response.status_code == 200:
            data = response.json()
            for item in data["response"]["results"]:
                if "webUrl" in item:
                    links.append(item["webUrl"])
                    if len(links) >= max_results:
                        logger.info(f"Reached max_results limit: {max_results}")
                        break
        else:
            logger.error(f"Error fetching data from Guardian API: {response.status_code} - {response.text}")
        
        if len(links) >= max_results:
            break

    logger.debug(f"Total links fetched: {len(links)}")
    return links[:max_results]
    

def get_news_text_from_links(links):
    texts = []

    for link in links:
        try:
            logger.info(f"Fetching {len(texts) + 1}/{len(links)}")
            response = requests.get(link)
            if response.status_code == 200:
                text = ""
                soup = BeautifulSoup(response.text, 'html.parser')

                standfirst_divs = soup.find_all('div', attrs={'data-gu-name': 'headline'})
                for div in standfirst_divs:
                    text += div.get_text(strip=False) + "\n"

                standfirst_divs = soup.find_all('div', attrs={'data-gu-name': 'standfirst'})
                for div in standfirst_divs:
                    text += div.get_text(strip=False) + "\n"

                standfirst_divs = soup.find_all('div', attrs={'data-gu-name': 'body'})
                for div in standfirst_divs:
                    text += div.get_text(strip=False) + "\n"

                texts.append(text.strip())
            else:
                logger.error(f"Error fetching content from {link}: {response.status_code}")
        except Exception as e:
            logger.error(f"Exception occurred while fetching content from {link}: {e}")
    
    logger.debug(f"Total texts fetched: {len(texts)}")
    return texts



if __name__ == "__main__":
    links = get_relevant_news_links("AI", {"from-date": "2023-01-01", "to-date": "2023-10-01", "page-size": 5, "order-by": "newest"}, max_results=5)
    texts = get_news_text_from_links(links)

    for text in texts:
        print(text)
        print("="*80)
        print("\n")