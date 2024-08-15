import os
import csv
import requests
from bs4 import BeautifulSoup
import asyncio

from realign.llm_utils import allm_messages_call


# File to store processed links
cwd = os.path.join(os.getcwd())
PROCESSED_LINKS_FILE = os.path.join(cwd, 'processed_links.txt')
BASE_CONTENT_REPO = os.path.join(cwd, 'base_content_repo.csv')
INFO_BITES_FILE = os.path.join(cwd, 'info_bites.csv')

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

async def generate_info_bites(agent, text, link):
    # if text is longer than 500,000 characters, truncate it to that length
    if len(text) > 200000:
        text = text[:200000]
    try:
        messages = [
            {"role": "user", "content": str(text)}
        ]
        
        response = await allm_messages_call(agent, messages)
        print('\n\n', '-' * 100)
        print(f'Agent: {agent}\nLink: {link}\nContent:\n\n', response.content)
        print('-' * 100, '\n\n')
        
        eval_response = await allm_messages_call('eval_agent', template_params={'messages': response.content})

        if eval_response.content["rating"] < 3:
            messages.append(response)
            new_message = {"role": "user",
                           "content": f"{eval_response.content['explanation']}\n\nCorrect your post based on the feedback above."}
            messages.append(new_message)
            
            response = await allm_messages_call(agent, messages)
            eval_response = await allm_messages_call('eval_agent', template_params={'messages': response.content})

        return response.content, eval_response.content
    
    except Exception as e:
        print(f"Error generating info bites: {e}")
        raise

async def process_link(agent, timestamp, text, link, processed_links, writer):
    if link and link not in processed_links:
        content = fetch_content(link)
        if content:
            info_bites, eval_response = await generate_info_bites(agent, content, link)
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
    # check if this file exists
    if os.path.exists(PROCESSED_LINKS_FILE):
        os.remove(PROCESSED_LINKS_FILE)

    with open(BASE_CONTENT_REPO, 'r', newline='', encoding='utf-8') as input_file, \
         open(INFO_BITES_FILE, 'a', newline='', encoding='utf-8') as output_file:

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
                for agent in ['twitter_content_agent', 'linkedin_content_agent']:
                    # Create a task for each link   
                    task = asyncio.create_task(process_link(agent, timestamp, text, link, processed_links, writer))
                    tasks.append(task)
            if len(tasks) > 0: break

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
