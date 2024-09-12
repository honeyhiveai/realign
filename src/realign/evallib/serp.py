from realign.evaluators import aevaluator, evaluator

import os
import httpx
from bs4 import BeautifulSoup


@aevaluator
async def google_flights(departure_id: str, arrival_id: str, outbound_date: str, return_date: str):

    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://serpapi.com/search",
            params={
                "engine": "google_flights",
                "api_key": os.getenv('SERP_API_KEY'),
                "departure_id": departure_id,
                "arrival_id": arrival_id,
                "outbound_date": outbound_date,
                "return_date": return_date,
            },
        )
        response_json = response.json()

    return response_json


@aevaluator
async def google_search(q: str):
    print(f'Searching Google for {q}')

    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://serpapi.com/search",
            params={
                "engine": "google",
                "api_key": os.getenv('SERP_API_KEY'),
                "q": q,
            },
        )
        response_json = response.json()
    
    return response_json

@aevaluator
async def get_website_text(url: str) -> dict:
    print(f'Getting text and links from {url}')

    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Get all links
        links = {}
        for a in soup.find_all('a', href=True):
            links[a.text.strip()] = a['href']
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)

    return {
        "text": text,
        "links": links
    }

@evaluator
def continue_thinking():
    return 'continuing'
