import os

from llama_index import download_loader
from llama_index.node_parser import SimpleNodeParser
from llama_index import GPTVectorStoreIndex

download_loader("GithubRepositoryReader")

from llama_index.readers.llamahub_modules.github_repo import (
    GithubRepositoryReader,
    GithubClient,
)

# Initialize the GithubRepositoryReader
github_client = GithubClient(os.getenv("GITHUB_TOKEN"))
loader = GithubRepositoryReader(
    github_client,
    owner="foundry-rs",
    repo="foundry",
    filter_directories=(
        ['.cargo', '.github', 'abi', 'anvil', 'binder',
         'cast', 'chisel', 'cli', 'common', 'config', 'doc',
         'docs', 'evm', 'fmt', 'forge', 'foundryup', 'macros',
         'testdata', 'ui', 'utils'],
        GithubRepositoryReader.FilterType.INCLUDE,
    ),
    filter_file_extensions=(['.md', '.sol', '.json', '.js', '.rs', '.toml', '.txt', '.yml']
                            , GithubRepositoryReader.FilterType.INCLUDE),
    verbose=True,
    concurrent_requests=10,
)

# 1. Load the documents
docs = loader.load_data(branch="master")

# 2. Parse the docs into nodes
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(docs)

# 3. Build an index
# You can customize the LLM. By default it uses `text-davinci-003`
index = GPTVectorStoreIndex(nodes)

# 4. Persist the index
index.storage_context.persist(persist_dir="index")
