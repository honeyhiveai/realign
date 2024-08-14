
import os
import csv
import requests
import json
from datetime import datetime

# Slack configuration
SLACK_API_TOKEN = os.getenv('SLACK_API_TOKEN')
SLACK_CHANNEL = 'C03LATH1D28'  # Replace with your desired channel ID

# Slack API endpoint
SLACK_API_URL = 'https://slack.com/api/conversations.history'

# Headers for API requests
headers = {
    'Authorization': f'Bearer {SLACK_API_TOKEN}',
    'Content-Type': 'application/json'
}

# get current working directory
cwd = os.path.join(os.getcwd(), 'examples', 'content_agent')
BASE_CONTENT_REPO = os.path.join(cwd, 'base_content_repo.csv')


# Function to extract links from message
def extract_links(message):
    links = []
    if 'attachments' in message:
        for attachment in message.get('attachments', []):
            if 'original_url' in attachment:
                links.append(attachment['original_url'])
    if 'blocks' in message:
        for block in message.get('blocks', []):
            if block['type'] == 'rich_text':
                for element in block.get('elements', []):
                    if element['type'] == 'rich_text_section':
                        for item in element.get('elements', []):
                            if item['type'] == 'link':
                                links.append(item['url'])
    return links

# Function to fetch messages and write to CSV
def fetch_and_write_messages():
    try:
        # Read existing entries
        existing_entries = set()
        try:
            with open(BASE_CONTENT_REPO, 'r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                for row in reader:
                    existing_entries.add((row[0], row[1], row[2]))
        except FileNotFoundError:
            pass  # File doesn't exist yet, which is fine

        # Prepare the API request payload
        payload = {
            'channel': SLACK_CHANNEL,
            'limit': 10 # Adjust this number based on how many messages you want to fetch
        }

        # Make the API request
        response = requests.post(SLACK_API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()

        if not data['ok']:
            raise Exception(f"Slack API error: {data['error']}")

        messages = data['messages']

        # Open CSV file for writing
        with open(BASE_CONTENT_REPO, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            # Write header if file is empty
            if file.tell() == 0:
                writer.writerow(['Timestamp', 'Text', 'Link'])

            new_entries_count = 0

            # Process each message
            for message in messages:
                text = message.get('text', '').strip()
                links = extract_links(message)
                timestamp = datetime.fromtimestamp(float(message['ts'])).strftime('%Y-%m-%d %H:%M:%S')

                # If there are no links, write the text with an empty link
                if not links:
                    entry = (timestamp, text, '')
                    if entry not in existing_entries:
                        writer.writerow(entry)
                        existing_entries.add(entry)
                        new_entries_count += 1
                else:
                    # Write a row for each link
                    for link in links:
                        entry = (timestamp, text, link)
                        if entry not in existing_entries:
                            writer.writerow(entry)
                            existing_entries.add(entry)
                            new_entries_count += 1

        print(f"{new_entries_count} new messages have been appended to base_content_repo.csv")

    except requests.exceptions.RequestException as e:
        print(f"Error making request to Slack API: {e}")
    except Exception as e:
        print(f"Error: {e}")

# Run the script
if __name__ == "__main__":
    fetch_and_write_messages()
