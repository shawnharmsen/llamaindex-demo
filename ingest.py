import os
import time
import pickle
import logging
from dotenv import load_dotenv
from contextlib import contextmanager
from multiprocessing import Pool

from llama_index import download_loader
from llama_index.node_parser import SimpleNodeParser
from llama_index import GPTVectorStoreIndex

PICKLE_DOCS_DIR = "pickled_docs"

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

if not os.path.exists(PICKLE_DOCS_DIR):
    os.makedirs(PICKLE_DOCS_DIR)

@contextmanager
def timer(msg):
    start = time.time()
    yield
    end = time.time()
    logging.info(f"{msg} took {end - start:.2f} seconds")

def load_pickle(filename):
    with open(os.path.join(PICKLE_DOCS_DIR, filename), "rb") as f:
        return pickle.load(f)

def save_pickle(obj, filename):
    with open(os.path.join(PICKLE_DOCS_DIR, filename), "wb") as f:
        pickle.dump(obj, f)

def parse_document(doc_tuple):
    idx, document = doc_tuple
    logging.info(f"Starting to parse document number: {idx}")
    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents([document])
    logging.info(f"Finished parsing document number: {idx}")
    return nodes


load_dotenv()
download_loader("GithubRepositoryReader")

from llama_index.readers.llamahub_modules.github_repo import (
    GithubRepositoryReader,
    GithubClient,
)

github_token = os.getenv("GITHUB_TOKEN")
openai_api_key = os.getenv("OPENAI_API_KEY")

github_client = GithubClient(github_token)
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
    filter_file_extensions=(['.md', '.sol', '.json', '.js', '.rs', '.toml', '.txt', '.yml'],
                            GithubRepositoryReader.FilterType.INCLUDE),
    verbose=True,
    concurrent_requests=10,
)

if __name__ == '__main__':
    try:
        pickled_docs_filename = "docs.pkl"
        if os.path.exists(os.path.join(PICKLE_DOCS_DIR, pickled_docs_filename)):
            with timer("Loading pickled documents"):
                docs = load_pickle(pickled_docs_filename)
        else:
            with timer("Loading documents"):
                docs = loader.load_data(branch="master")
            save_pickle(docs, pickled_docs_filename)

        logging.info("Starting to parse documents")
        with timer("Parsing documents"):
            with Pool() as p:
                nodes_list = p.map(parse_document, enumerate(docs)) # use enumerate to get the index of each document
            nodes = [node for sublist in nodes_list for node in sublist]
        logging.info("Finished parsing documents")

        with timer("Building index"):
            index = GPTVectorStoreIndex(nodes)

        with timer("Persisting index"):
            index.storage_context.persist(persist_dir="index")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
