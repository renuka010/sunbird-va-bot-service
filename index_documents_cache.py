import json
from typing import (
    List
)
from env_manager import vectorstore_class
from langchain.docstore.document import Document
from utils.env import get_from_env_or_config
 
def store_cache_in_marqo(query: str, contexts: str, context:str):
    document = Document(page_content=query,
                          metadata={
                              "response": contexts
                            })
    indices_cached = json.loads(get_from_env_or_config('database', 'indices_cached', None))
    index_id = indices_cached.get(context.lower())
    print("Adding documents to cache...")
    result = vectorstore_class.cache_documents(document, index_id)
    print("result =======>", result)
    print("============ CACHING INDEX DONE =============")

def search_cache_in_marqo(query: str, context: str) -> str:
    indices_cached = json.loads(get_from_env_or_config('database', 'indices_cached', None))
    index_id = indices_cached.get(context.lower())
    documents = vectorstore_class.similarity_search_with_score(query, index_id)
    if documents:
        top_match = documents[0]
        cache_min_score: float = float(get_from_env_or_config('database', 'cache_min_score', None))
        if float(top_match[1]) >= cache_min_score:
            print("========== Response retrieved from cache ==========")
            return top_match[0].metadata.get('response')
    return None


