import os
import time
import pickle
import logging
import tempfile
import itertools
from multiprocessing import Pool
from contextlib import contextmanager
from dotenv import load_dotenv
from llama_index import GPTVectorStoreIndex, download_loader
from llama_index.readers.llamahub_modules.github_repo import GithubRepositoryReader, GithubClient
from llama_index.node_parser import SimpleNodeParser

BATCH_SIZE = 50
PICKLE_DOCS_DIR = "pickled_docs"
os.makedirs(PICKLE_DOCS_DIR, exist_ok=True)
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
    with tempfile.NamedTemporaryFile(dir=PICKLE_DOCS_DIR, delete=False) as f:
        pickle.dump(obj, f)
        temp_filepath = f.name
    os.rename(temp_filepath, final_filepath)

def parse_documents_wlog(batch):
    idx, documents = batch
    logging.info(f"Starting to parse batch number: {idx}")
    nodes = SimpleNodeParser().get_nodes_from_documents(documents)
    logging.info(f"Finished parsing batch number: {idx}")
    save_pickle(nodes, f"nodes_batch_{idx}.pkl")
    return idx

load_dotenv()
download_loader("GithubRepositoryReader")
github_token, openai_api_key  = os.getenv("GITHUB_TOKEN"), os.getenv("OPENAI_API_KEY")
github_client = GithubClient(github_token)

loader = GithubRepositoryReader(
    github_client,
    owner="foundry-rs",
    repo="foundry",
    filter_directories=(
        ['.cargo', '.github', 'abi', 'anvil', 'binder', 'cast', 'chisel', 'cli', 'common', 'config', 'doc', 'docs', 'evm', 'fmt', 'forge', 'foundryup', 'macros', 'testdata', 'ui', 'utils'],
        GithubRepositoryReader.FilterType.INCLUDE,
    ),
    filter_file_extensions=(['.md', '.sol', '.json', '.js', '.rs', '.toml', '.txt', '.yml'], GithubRepositoryReader.FilterType.INCLUDE),
    verbose=True,
    concurrent_requests=10)

if __name__ == '__main__':
    pickled_docs_fp = get_pickle_path("docs.pkl")
    docs = (load_pickle("docs.pkl") if os.path.exists(pickled_docs_fp)
            else loader.load_data(branch="master"))

    if not os.path.exists(pickled_docs_fp):
        with timer("Saving pickled documents"):
            save_pickle(docs, "docs.pkl")

    doc_batches = [(i, docs[i: i + BATCH_SIZE]) for i in range(0, len(docs), BATCH_SIZE)]
    pickled_files = {f for f in os.listdir(PICKLE_DOCS_DIR) if os.path.isfile(get_pickle_path(f))}
    unparsed_batches = [batch for batch in doc_batches if f"nodes_batch_{batch[0]}.pkl" not in pickled_files]

    with timer("Parsing documents"):
        with Pool() as p:
            try:
                completed_batches = p.imap_unordered(parse_documents_wlog, unparsed_batches)
                for _ in itertools.repeat(None, len(unparsed_batches)):
                    logging.info(f"Batch {next(completed_batches)} has been pooled.")
            except KeyboardInterrupt:
                logging.info("Interrupt received, terminating processes")
                p.terminate()
                p.join()

    nodes = [load_pickle(f"nodes_batch_{i}.pkl") for i in range(len(doc_batches)) if f"nodes_batch_{i}.pkl" in pickled_files]
    nodes = [item for sublist in nodes for item in sublist]

    with timer("Building index"):
        index = GPTVectorStoreIndex(nodes)

    with timer("Persisting index"):
        index.storage_context.persist(persist_dir="index")
