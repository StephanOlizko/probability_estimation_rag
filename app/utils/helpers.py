import requests
import json
from os import environ
from dotenv import load_dotenv
import logging
from bs4 import BeautifulSoup
from groq import Groq, AsyncGroq
import asyncio
import aiohttp
from .prompts import PromptFactory
import random

# Configure logger
logger = logging.getLogger(__name__)

load_dotenv()


class OpenRouterClient:
    def __init__(self, model="deepseek/deepseek-chat-v3-0324:free"):
        logger.debug(f"Initializing OpenRouterClient with model: {model}")
        api_key = environ.get("API_KEY")
        if not api_key:
            logger.warning("API_KEY environment variable not found for OpenRouterClient")

        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = model

    def generate_response(self, messages=None, response_format=None):
        logger.debug(f"Generating response with OpenRouterClient using model: {self.model}")
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
            logger.debug("Sending request to OpenRouter API")
            response = requests.post(url, headers=headers, data=json.dumps(data))
            return response.json()
        
        except Exception as e:
            logger.error(f"Error in OpenRouterClient: {e}")
            return None    


class GroqClient:
    def __init__(self, model="llama-3.1-8b-instant"):
        logger.debug(f"Initializing GroqClient with model: {model}")
        api_key = environ.get("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY environment variable not found for GroqClient")

        self.model = model
        self.client = Groq(api_key=api_key, max_retries=5)

    def generate_response(self, messages=None, response_format={"type": "json_object"}, temperature=1, max_tokens=1024, top_p=1):
        logger.debug(f"Generating response with GroqClient using model: {self.model}")
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format=response_format,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                top_p=top_p,
                stream=False,
                stop=None,
            )
            logger.debug("Successfully received response from Groq API")
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in GroqClient: {e}")
            return None


class AsyncGroqClient:
    def __init__(self, model="llama-3.1-8b-instant"):
        logger.debug(f"Initializing AsyncGroqClient with model: {model}")
        api_key = environ.get("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY environment variable not found for AsyncGroqClient")

        self.model = model
        self.client = AsyncGroq(api_key=api_key, max_retries=5)

    async def generate_response(self, messages=None, response_format={"type": "json_object"}, temperature=1, max_tokens=1024, top_p=1):
        logger.debug(f"Generating async response with AsyncGroqClient using model: {self.model}")
        await asyncio.sleep(random.randint(5, 10))
        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format=response_format,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                top_p=top_p,
                stream=False,
                stop=None,
            )
            logger.debug("Successfully received response from async Groq API")
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in AsyncGroqClient: {e}")
            return None


def get_general_topic(text, client = None):

    client = GroqClient(model="llama-3.1-8b-instant")
    logger.info(f"Getting general topic using model: {client.model}")

    messages = PromptFactory.create_general_topic_prompt(text)
    response = client.generate_response(messages=messages, response_format={"type": "text"})

    return response
    

def generate_response_based_on_context(messages, model="deepseek/deepseek-chat-v3-0324:free"):
    logger.info(f"Generating response based on context using model: {model}")
    # This function is used to get the response based on the context using the OpenRouter API.
    # It takes the messages and model as input and returns the response.

    client = OpenRouterClient(model=model)
    response = client.generate_response(messages=messages)
    return response
    
def get_relevant_news_links(query, params=None, max_results=500):
    logger.info(f"Getting relevant news links for query: '{query}' with max_results: {max_results}")

    links = []

    base_url = "https://content.guardianapis.com/search"
    api_key = environ.get("GUARDIAN_API_KEY")

    if not api_key:
        logger.error("GUARDIAN_API_KEY is not set in the environment variables.")
        return []

    str_add = []
    if params is not None:
        logger.debug(f"Using search parameters: {params}")
        for key, value in params.items():
            if key in ["from-date", "to-date", "page-size", "order-by"]:
                str_add.append(f"{key}={value}")

    str_add.append("page-size=100")
    str_add.append("order-by=relevance")

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

    logger.info(f"Total links fetched: {len(links)}")
    return list(reversed(links[:max_results]))
    

def get_news_text_from_links(links):
    logger.info(f"Getting news text from {len(links)} links (synchronous)")
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
                logger.debug(f"Successfully fetched content from {link}")
            else:
                logger.error(f"Error fetching content from {link}: {response.status_code}")
        except Exception as e:
            logger.error(f"Exception occurred while fetching content from {link}: {e}")
    
    logger.info(f"Total texts fetched: {len(texts)}")
    return texts


async def get_news_text_from_links_async(links):
    logger.info(f"Getting news text from {len(links)} links (asynchronous)")
    texts = []
    
    async def fetch_link(link, retry_count=10, backoff_factor=2.5):
        for attempt in range(retry_count + 1):
            try:
                delay = 0.5 * (backoff_factor ** attempt) + 0.3
                if attempt >= 0:
                    logger.info(f"Retry attempt {attempt} for {link} with delay {delay:.2f}s")
                    await asyncio.sleep(delay)

                session = aiohttp.ClientSession()
                try:
                    logger.info(f"Fetching link {links.index(link) + 1}/{len(links)}")
                    async with session.get(link) as response:
                        if response.status == 200:
                            text = ""
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')
                            
                            standfirst_divs = soup.find_all('div', attrs={'data-gu-name': 'headline'})
                            for div in standfirst_divs:
                                text += div.get_text(strip=False) + "\n"
                            
                            standfirst_divs = soup.find_all('div', attrs={'data-gu-name': 'standfirst'})
                            for div in standfirst_divs:
                                text += div.get_text(strip=False) + "\n"
                            
                            standfirst_divs = soup.find_all('div', attrs={'data-gu-name': 'body'})
                            for div in standfirst_divs:
                                text += div.get_text(strip=False) + "\n"
                            
                            logger.debug(f"Successfully fetched content from {link}")
                            return text.strip()
                        else:
                            logger.error(f"Error fetching content from {link}: {response.status}")
                            return None
                finally:
                    await session.close()
            except Exception as e:
                logger.error(f"Exception occurred while fetching content from {link}: {e}")
                if attempt == retry_count:
                    return None
    
    # Create tasks for all links
    logger.debug(f"Creating tasks for {len(links)} links")
    tasks = [fetch_link(link) for link in links]
    
    # Execute all tasks concurrently
    logger.debug("Executing tasks concurrently")
    results = await asyncio.gather(*tasks)
    
    # Filter out None results
    texts = [text for text in results if text is not None]
    
    logger.info(f"Total texts fetched asynchronously: {len(texts)}")
    return texts


if __name__ == "__main__":
    # Set up logging for main execution
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    import time
    logger.info("Starting news retrieval process")
    
    links = get_relevant_news_links("Ukraine", max_results=100)
    logger.info(f"Fetched {len(links)} links")

    logger.info("Starting synchronous fetching")
    t = time.time()
    texts = get_news_text_from_links(links)
    elapsed = time.time() - t
    logger.info(f"Time taken for synchronous fetching: {elapsed:.2f} seconds")
    logger.info(f"Fetched {len(texts)} texts")

    logger.info("_" * 50)

    logger.info("Starting asynchronous fetching")
    t = time.time()
    loop = asyncio.get_event_loop()
    texts = loop.run_until_complete(get_news_text_from_links_async(links))
    elapsed = time.time() - t
    logger.info(f"Time taken for asynchronous fetching: {elapsed:.2f} seconds")
    logger.info(f"Fetched {len(texts)} texts")