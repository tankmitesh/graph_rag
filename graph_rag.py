import os
import sys
import nltk
from dotenv import load_dotenv
from typing import List

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

# Local imports
from context_navigator import QueryEngine
from data_splitter import DocumentProcessor
from semantic_mapper import KnowledgeGraph

# Load environment variables
load_dotenv()

# Set environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Download NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)


class GraphRAG:
    """Graph-based Retrieval-Augmented Generation (RAG) system."""

    def __init__(self, input_documents: List[str]):
        """Initializes the GraphRAG system.

        Args:
            input_documents (List[str]): A list of documents to be processed.

        Attributes:
            language_model: An instance of a large language model (LLM) for generating responses.
            embedding_tool: An instance of an embedding model for document embeddings.
            doc_processor: An instance of the DocumentProcessor class for processing documents.
            knowledge_graph_tool: An instance of the KnowledgeGraph class for building and managing the knowledge graph.
            query_handler: An instance of the QueryEngine class for handling queries (initialized as None).
            graph_visualizer: An instance of the Visualizer class for visualizing the knowledge graph traversal.
        """
        self.language_model = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=4000)
        self.embedding_tool = OpenAIEmbeddings()
        self.doc_processor = DocumentProcessor()
        self.knowledge_graph_tool = KnowledgeGraph()
        self.query_handler = None
        self.graph_visualizer = Visualizer()
        self.process_documents(input_documents)

    def process_documents(self, input_documents: List[str]):
        """Processes documents by splitting, embedding, and building a knowledge graph.

        Args:
            input_documents (List[str]): A list of documents to be processed.

        Returns:
            None
        """
        doc_splits, vector_storage = self.doc_processor.process_documents(input_documents)
        self.knowledge_graph_tool.build_graph(doc_splits, self.language_model, self.embedding_tool)
        self.query_handler = QueryEngine(vector_storage, self.knowledge_graph_tool, self.language_model)

    def query(self, user_query: str) -> str:
        """Handles a query by retrieving information and visualizing the traversal path.

        Args:
            user_query (str): The query to be answered.

        Returns:
            str: The response to the query.
        """
        response, traversal_path, filtered_content = self.query_handler.query(user_query)

        return response


