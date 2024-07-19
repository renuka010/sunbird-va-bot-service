import json
from typing import (
    List
)
from env_manager import vectorstore_class
from langchain.docstore.document import Document
from utils.env import get_from_env_or_config
from utils.utils import prepare_redis_cache_key
from redis_util import store_response_in_redis, read_response_from_redis
from faiss_indexer import FaissIndexer
 
def store_response_in_cache(query: str, contexts: str, context:str):
    document = Document(page_content=query,
                          metadata={
                              "response": contexts
                            })
    indices_cached = json.loads(get_from_env_or_config('database', 'indices_cached', None))
    index_id = indices_cached.get(context.lower())
    print("Adding documents to cache...")
    result = vectorstore_class.cache_documents(document, index_id)
    print("result =======>", result)
    print("============ MARQO CACHING INDEX DONE =============")

    redis_key = prepare_redis_cache_key(context, query)
    store_response_in_redis(redis_key, contexts)
    print("============ REDIS CACHING INDEX DONE =============")

    faiss_index = FaissIndexer.load_index(index_id)
    if not faiss_index:
        faiss_index = FaissIndexer()
        faiss_index.build_index()
    faiss_index.store_query_in_faiss(query)
    faiss_index.save_index(index_id)
    print("============ FAISS CACHING INDEX DONE =============")

def search_query_in_cache(query: str, context: str) -> str:
    indices_cached = json.loads(get_from_env_or_config('database', 'indices_cached', None))
    index_id = indices_cached.get(context.lower())

    faiss_index = FaissIndexer.load_index(index_id)
    if not faiss_index:
        return None
    D, I, search_result = faiss_index.search_index(query)
    score = float(D[0])
    cache_max_score: float = float(get_from_env_or_config('database', 'cache_max_score', None))
    if score <= cache_max_score:
        redis_key = prepare_redis_cache_key(context, search_result)
        response = read_response_from_redis(redis_key)
        print("========== Response retrieved from cache ==========")
        return response
    return None
    


