import requests
import os

def get_file_extensions(user, repo):
    api_url = f"https://api.github.com/repos/{user}/{repo}/git/trees/master?recursive=1"  # changed main to master
    headers = {"Accept": "application/vnd.github+json"}

    response = requests.get(api_url, headers=headers)
    response.raise_for_status()  # Raise an exception if the request failed

    data = response.json()
    extensions = set()  # use a set to automatically discard duplicates

    for file in data['tree']:
        _, extension = os.path.splitext(file['path'])
        if extension:  # if the file has an extension
            extensions.add(extension)

    return list(extensions)  # convert set to list

# Usage
user = "foundry-rs"
repo = "foundry"
print(get_file_extensions(user, repo))
