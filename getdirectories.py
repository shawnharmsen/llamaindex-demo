import requests
import os

def get_top_level_directories(user, repo):
    api_url = f"https://api.github.com/repos/{user}/{repo}/contents"  # Only get the top level content
    headers = {"Accept": "application/vnd.github+json"}

    response = requests.get(api_url, headers=headers)
    response.raise_for_status()  # Raise an exception if the request failed

    data = response.json()
    directories = []  # use a list to store directories

    for item in data:
        if item['type'] == 'dir':  # if the item is a directory
            directories.append(item['name'])

    return directories  # return the list of directories

# Usage
user = "foundry-rs"
repo = "foundry"
print(get_top_level_directories(user, repo))
