import os
import json
import csv
import requests
from bs4 import BeautifulSoup
import asyncio
from realign.llm_utils import allm_messages_call
from realign.types import ModelSettings, OpenAIMessage

# File to store processed links
PROCESSED_LINKS_FILE = 'processed_links.txt'

INFO_SYSTEM_PROMPT = open('info_prompt.txt', 'r').read()

eval_model_settings = ModelSettings(
    model='openai/gpt-4o-mini',
    template='rating_5_star',
    json_mode=True,
    prompt_params = {
        'criteria': '\n'.join([
            "1. Make sure sentences are concise and don't use flowery language.",
            "2. It shouldn't sound too salesy or promotional.",
            "3. Don't use too many adjectives or adverbs.",
            "4. Don't make general claims.",
            "5. Don't start with a question."
        ])
    }
)

base_model_settings = ModelSettings(
    model='anthropic/claude-3-sonnet-20240229',
    template=INFO_SYSTEM_PROMPT,
    hyperparams = {
        'max_tokens': 1024,
    }
)
base_model_settings.prompt_params = {
    'platform': 'Twitter',
    'platform_instructions': """
Make sure the tweet is around 600-800 characters long.
""",
    'example': """
Mixture-of-Agents as an Event-Driven System 🤖☎️

This paper by @JunlinWang3 shows you how to ensemble smaller LLMs to create a system that can outperform state-of-the-art larger models.

We’ve implemented this paper in a fully async, event-driven workflow system thanks to @ravithejads
- treat each “small LLM” as an event-driven step that can process incoming events and respond to it, independently and in parallel.

✅ Take full advantage of processing an entire batch of requests
✅ Cleaner, readable code

LlamaPack: https://llamahub.ai/l/llama-packs/llama-index-packs-mixture-of-agents?from=
Learn more about workflows: https://llamaindex.ai/blog/introducing-workflows-beta-a-new-way-to-create-complex-ai-applications-with-llamaindex
"""
}

print(base_model_settings.resolve_system_prompt())
linkedin_model_settings = base_model_settings.copy()
linkedin_model_settings.prompt_params = {
    'platform': 'LinkedIn',
    'platform_instructions': """
Make sure the tone is professional and engaging.
Don't exaggerate any claims.
Don't sound like a marketer or a salesperson. Sound like an intelligent investor.
""",
    'example': """
Ever wonder why your LLM app aces your test suite but stumbles in production? You might be seeing dataset drift.

Real-world usage is dynamic. User inputs evolve, model behavior changes (remember those unexpected OpenAI updates?), and user expectations shift. Meanwhile, our golden datasets often remain frozen in time. The result? "Dataset Drift" - where your test cases no longer represent real user queries.

The solution? Build a data flywheel 🔄

1️⃣ Continuously monitor and evaluate logs from production (using feedback, auto-evals, etc.)
2️⃣ Use metrics and scores to filter your logs and find underperforming queries
3️⃣ Add these edge-cases to your test bank and manually correct LLM outputs
😎 Watch your test suite evolve with users

We’re seeing teams significantly improve LLM reliability by adopting this approach. Remember: Both your datasets and eval criteria need to continuously evolve with real-world usage!

Slides from my OSS4AI talk on this topic: https://lnkd.in/eHjhkkV9
"""
}

def load_processed_links():
    if os.path.exists(PROCESSED_LINKS_FILE):
        with open(PROCESSED_LINKS_FILE, 'r') as f:
            return set(line.strip() for line in f)
    return set()

def save_processed_link(link):
    with open(PROCESSED_LINKS_FILE, 'a') as f:
        f.write(f"{link}\n")

def fetch_content(url):
    # twitter and pdf parsing are broken
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
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
        
        return text
    except Exception as e:
        print(f"Error fetching content from {url}: {e}")
        return None

async def generate_info_bites(model_settings, text):
    # if text is longer than 500,000 characters, truncate it to that length
    if len(text) > 200000:
        text = text[:200000]
    try:
        messages = [
            OpenAIMessage(
                role="user",
                content=f"{text}"
            )
        ]
        response = await allm_messages_call(model_settings, messages)
        eval_model_settings.prompt_params["messages"] = response.content
        eval_response = await allm_messages_call(eval_model_settings)
        print(response.content)
        print("eval score", eval_response.content)
        if eval_response.content["rating"] < 3:
            messages.append(response)
            new_message = OpenAIMessage(
                role="user",
                content=f"{eval_response.content["explanation"]}\n\nCorrect your post based on the feedback above."
            )
            messages.append(new_message)
            response = await allm_messages_call(model_settings, messages)
            eval_model_settings.prompt_params["messages"] = response.content
            eval_response = await allm_messages_call(eval_model_settings)
        return response.content, eval_response.content
    except Exception as e:
        print(f"Error generating info bites: {e}")
        return None

async def process_link(model_settings, timestamp, text, link, processed_links, writer):
    if link and link not in processed_links:
        content = fetch_content(link)
        if content:
            info_bites, eval_response = await generate_info_bites(model_settings, content)
            if info_bites:
                info_bites_combined = ' | '.join([bite.strip() for bite in info_bites.split('\n') if bite.strip()])
                writer.writerow([timestamp, text, link, info_bites_combined, eval_response['explanation'], eval_response['rating']])
                processed_links.add(link)
                save_processed_link(link)
            else:
                writer.writerow([timestamp, text, link, "Failed to generate info bites"])
        else:
            writer.writerow([timestamp, text, link, "Failed to fetch content"])
    elif link in processed_links:
        print(f"Skipping already processed link: {link}")
    else:
        writer.writerow([timestamp, text, link, "No link provided"])

async def process_csv():
    with open('base_content_repo.csv', 'r', newline='', encoding='utf-8') as input_file, \
         open('info_bites.csv', 'a', newline='', encoding='utf-8') as output_file:

        processed_links = load_processed_links()
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)
        if output_file.tell() == 0:
            writer.writerow(['Timestamp', 'Original Text', 'Link', 'Info Bites', 'Eval Explanation', 'Eval Rating'])
        
        # Skip header
        next(reader)

        # Create a list to store all the tasks
        tasks = []

        for row in reader:
            timestamp, text, link = row
            if "x.com" not in link:
                for model_settings in [base_model_settings, linkedin_model_settings]:
                    # Create a task for each link   
                    task = asyncio.create_task(process_link(model_settings, timestamp, text, link, processed_links, writer))
                    tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
