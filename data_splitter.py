# Third Party Packages
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings
import numpy as np


class DocumentProcessor:
    def __init__(self):
        """
        Initializes the DocumentProcessor with a text splitter and OpenAI embeddings.

        Attributes:
        - text_chunk_splitter: An instance of RecursiveCharacterTextSplitter with specified chunk size and overlap.
        - embedding_generator: An instance of OpenAIEmbeddings used for embedding documents.
        """
        self.text_chunk_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embedding_generator = OpenAIEmbeddings()

    def process_documents(self, input_docs):
        """
        Processes a list of documents by splitting them into smaller chunks and creating a vector store.

        Args:
        - input_docs (list of str): A list of documents to be processed.

        Returns:
        - tuple: A tuple containing:
          - doc_splits (list of str): The list of split document chunks.
          - vector_storage (FAISS): A FAISS vector store created from the split document chunks and their embeddings.
        """
        doc_splits = self.text_chunk_splitter.split_documents(input_docs)
        vector_storage = FAISS.from_documents(doc_splits, self.embedding_generator)
        return doc_splits, vector_storage

    def create_embeddings_batch(self, texts, batch_size=32):
        """
        Creates embeddings for a list of texts in batches.

        Args:
        - texts (list of str): A list of texts to be embedded.
        - batch_size (int, optional): The number of texts to process in each batch. Default is 32.

        Returns:
        - numpy.ndarray: An array of embeddings for the input texts.
        """
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedding_generator.embed_documents(batch)
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)

    def compute_similarity_matrix(self, embeddings):
        """
        Computes a cosine similarity matrix for a given set of embeddings.

        Args:
        - embeddings (numpy.ndarray): An array of embeddings.

        Returns:
        - numpy.ndarray: A cosine similarity matrix for the input embeddings.
        """
        return cosine_similarity(embeddings)