import os
import time
import pickle
import logging
import tempfile
from multiprocessing import Pool
from contextlib import contextmanager
from dotenv import load_dotenv

from llama_index import download_loader
from llama_index.node_parser import SimpleNodeParser
from llama_index import GPTVectorStoreIndex

BATCH_SIZE = 100  # Size of batches for batch processing
PICKLE_DOCS_DIR = "pickled_docs"

# Create the directory for pickled docs if it does not exist
os.makedirs(PICKLE_DOCS_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

@contextmanager
def timer(msg):
    start = time.time()
    yield
    end = time.time()
    logging.info(f"{msg} took {end - start:.2f} seconds")

def get_pickle_path(filename):
    return os.path.join(PICKLE_DOCS_DIR, filename)

def load_pickle(filename):
    with open(get_pickle_path(filename), "rb") as f:
        return pickle.load(f)

def save_pickle(obj, filename):
    final_filepath = get_pickle_path(filename)
    # Use NamedTemporaryFile for atomic write
    with tempfile.NamedTemporaryFile(dir=PICKLE_DOCS_DIR, delete=False) as f:
        pickle.dump(obj, f)
        temp_filepath = f.name
    os.rename(temp_filepath, final_filepath)

def parse_documents(batch):
    idx, documents = batch
    logging.info(f"Starting to parse batch number: {idx}")
    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    logging.info(f"Finished parsing batch number: {idx}")
    # Save parsed nodes
    save_pickle(nodes, f"nodes_batch_{idx}.pkl")

load_dotenv()
download_loader("GithubRepositoryReader")

from llama_index.readers.llamahub_modules.github_repo import GithubRepositoryReader, GithubClient

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
    pickled_docs_filename = "docs.pkl"
    pickled_docs_filepath = get_pickle_path(pickled_docs_filename)
    if os.path.exists(pickled_docs_filepath):
        with timer("Loading pickled documents"):
            docs = load_pickle(pickled_docs_filename)
    else:
        with timer("Loading documents"):
            docs = loader.load_data(branch="master")
        with timer("Saving pickled documents"):
            save_pickle(docs, pickled_docs_filename)

    # Split documents into batches
    doc_batches = [(i, docs[i: i + BATCH_SIZE]) for i in range(0, len(docs), BATCH_SIZE)]

    # Check which batches have already been parsed
    pickled_files = {f for f in os.listdir(PICKLE_DOCS_DIR) if os.path.isfile(get_pickle_path(f))}
    unparsed_batches = [batch for batch in doc_batches if f"nodes_batch_{batch[0]}.pkl" not in pickled_files]

    with timer("Parsing documents"):
        try:
            with Pool() as p:
                p.map(parse_documents, unparsed_batches)
        except KeyboardInterrupt:
            # Handle interrupt during parsing
            logging.info("Interrupt received, terminating processes")
            p.terminate()
            p.join()
        finally:
            # Clean up resources if necessary
            pass

    # Collect nodes from the parsed batches
    nodes = []
    for i in range(len(doc_batches)):
        batch_filename = f"nodes_batch_{i}.pkl"
        if batch_filename in pickled_files:
            nodes.extend(load_pickle(batch_filename))

    with timer("Building index"):
        index = GPTVectorStoreIndex(nodes)

    with timer("Persisting index"):
        index.storage_context.persist(persist_dir="index")
