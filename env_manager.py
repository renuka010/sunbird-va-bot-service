import os
from dotenv import load_dotenv
from logger import logger

from translation import (
    BaseTranslationClass,
    BhashiniTranslationClass,
    DhruvaTranslationClass,
    GoogleCloudTranslationClass
)
from storage import (
    BaseStorageClass,
    AwsS3BucketClass,
    GcpBucketClass,
    OciBucketClass
)
from llm import (
    BaseChatClient,
    AzureChatClient,
    OpenAIChatClient,
    OllamaChatClient
)

from vectorstores import (
    BaseVectorStore,
    MarqoVectorStore
)


class EnvironmentManager():
    """
    Class for initializing functions respective to the env variable provided
    """

    def __init__(self):
        load_dotenv()
        self.indexes = {
            "llm": {
                "class": {
                    "openai": OpenAIChatClient,
                    "azure": AzureChatClient,
                    "ollama": OllamaChatClient
                },
                "env_key": "LLM_TYPE"
            },
            "translate": {
                "class": {
                    "bhashini": BhashiniTranslationClass,
                    "google": GoogleCloudTranslationClass,
                    "dhruva": DhruvaTranslationClass
                },
                "env_key": "TRANSLATION_TYPE"
            },
            "storage": {
                "class": {
                    "oci": OciBucketClass,
                    "gcp": GcpBucketClass,
                    "aws": AwsS3BucketClass
                },
                "env_key": "BUCKET_TYPE"
            },
            "vectorstore": {
                "class": {
                    "marqo": MarqoVectorStore
                },
                "env_key": "VECTOR_STORE_TYPE"
            }
        }

    def create_instance(self, env_key):
        env_var = self.indexes[env_key]["env_key"]
        type_value = os.getenv(env_var)

        if type_value is None:
            raise ValueError(
                f"Missing credentials. Please pass the `{env_var}` environment variable"
            )

        logger.info(f"Init {env_key} class for: {type_value}")
        return self.indexes[env_key]["class"].get(type_value)()


env_class = EnvironmentManager()

# create instances of functions
logger.info(f"Initializing required classes for components")
llm_class: BaseChatClient = env_class.create_instance("llm")


if os.environ["TRANSLATION_TYPE"] is None or os.environ["TRANSLATION_TYPE"] == "":
    logger.info("No translation class specified.")
    translate_class: BaseTranslationClass = None
else:
    translate_class: BaseTranslationClass = env_class.create_instance("translate")

if os.environ["BUCKET_TYPE"] is None or os.environ["BUCKET_TYPE"] == "":
    logger.info("No storage class specified.")
    storage_class: BaseStorageClass = None
else:
    storage_class: BaseStorageClass = env_class.create_instance("storage")

vectorstore_class: BaseVectorStore = env_class.create_instance("vectorstore")
