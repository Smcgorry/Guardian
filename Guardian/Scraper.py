import json
import os
import requests
from scrapegraphai.graphs import SearchGraph
from urllib.parse import urlparse
from whoosh.fields import Schema, TEXT, ID
from whoosh.index import open_dir, create_in
import os
from bs4 import BeautifulSoup

# Configuration for OpenAI model
graph_config = {
    "llm": {
        "model": "gpt-3.5-turbo",
        "temperature": 0,
        "api_key": "sk-proj-2OA94ojphMfgksXvav85T3BlbkFJXWdxppTapa6ALK3KmwcQ",
        "streaming": True,
    }
}

def normalize_keys(result):
    """Normalize result keys to unify them."""
    normalized_result = {}
    for key, value in result.items():
        new_key = 'urls' if 'urls' in key else key
        normalized_result[new_key] = value
    return normalized_result

def ensure_scheme(urls):
    """Ensure every URL starts with http:// or https://."""
    updated_urls = []
    for url in urls:
        if not url.startswith('http://') and not url.startswith('https://'):
            url = 'https://' + url
        updated_urls.append(url)
    return updated_urls

def check_url_accessibility(url):
    """Check if the URL is accessible."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"Failed to access {url}: {str(e)}")
        return False

def fetch_html(url):
    """Fetch the HTML content of a given URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Failed to fetch {url}: {str(e)}")
        return None

def sanitize_url(url):
    """Sanitize the URL to create a valid filename."""
    parsed_url = urlparse(url)
    filename = parsed_url.netloc + parsed_url.path
    filename = filename.replace('/', '_').replace('.', '_').strip('_') + '.html'
    return filename

def save_html(base_path, url, html_content):
    """Save the HTML content to a file named after the URL within a specified base path."""
    filename = sanitize_url(url)
    file_path = os.path.join(base_path, filename)
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(html_content)
        print(f"Saved HTML content for {url} in {file_path}")
    else:
        print(f"File already exists for {url}, skipping download.")

def read_existing_urls(file_path):
    """Read existing URLs from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_new_urls(file_path, new_urls):
    """Save new URLs to a JSON file, combining them with existing ones."""
    existing_urls = read_existing_urls(file_path)
    updated_urls = list(set(existing_urls + new_urls))
    with open(file_path, 'w') as file:
        json.dump(updated_urls, file, indent=2)

def add_html_to_index(index_dir, html_files_dir):
    ix = open_dir(index_dir)
    writer = ix.writer()
    # Loop through all HTML files in the directory
    for filename in os.listdir(html_files_dir):
        if filename.endswith(".html"):
            filepath = os.path.join(html_files_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'html.parser')
                # Extract title and content or other elements as needed
                title = soup.title.text if soup.title else 'No Title'
                content = soup.get_text()
                # Add document to index
                writer.add_document(title=title, content=content, path=filepath)
    writer.commit()

def setup_whoosh_index(index_path, schema):
    if not os.path.exists(index_path):
        os.mkdir(index_path)
        return create_in(index_path, schema)
    else:
        return open_dir(index_path)

def main():
    # Part 1: Retrieve and Normalize URLs
    file_path1 = "/home/headquarters/Documents/Guardian/FunctionRecs/WebDocs/URLs.json"
    prompt = input('Keyword to search URLs for: ')
    search_graph = SearchGraph(
        prompt=f"List all urls related to {prompt}",
        config=graph_config
    )
    result = search_graph.run()
    normalized_result = normalize_keys(result)
    urls = normalized_result.get('urls', [])
    urls_with_schemes = ensure_scheme(normalized_result.get('urls', []))

    with open(file_path1, 'w') as file:
        json.dump(urls, file, indent=2)

    accessible_urls = [url for url in urls_with_schemes if check_url_accessibility(url)]

    # Save accessible URLs
    file_path2 = "/home/headquarters/Documents/Guardian/FunctionRecs/WebDocs/AccessibleURLs.json"
    save_new_urls(file_path2, accessible_urls)

    # Part 2: Fetch and Save HTML content
    base_path = "/home/headquarters/Documents/Guardian/FunctionRecs/HTML/"
    for url in accessible_urls:
        html_content = fetch_html(url)
        if html_content:
            save_html(base_path, url, html_content)

    whoosh_schema = Schema(
    title=TEXT(stored=True),
    path=ID(stored=True, unique=True),
    content=TEXT(stored=True)
    )

    whoosh_index_path = "/home/headquarters/Documents/Guardian/WebIndex/"
    setup_whoosh_index(whoosh_index_path, whoosh_schema)
    add_html_to_index(whoosh_index_path, base_path)
if __name__ == "__main__":
    main()