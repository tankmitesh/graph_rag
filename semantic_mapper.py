import networkx as nx
from nltk.stem import WordNetLemmatizer
from spacy.cli import download
from langchain.prompts import PromptTemplate
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from typing import List, Any
from concepts import Concepts




class KnowledgeGraph:
    """Represents a knowledge graph built from document chunks.

    Attributes:
        graph_structure (networkx.Graph): Represents the knowledge graph structure.
        word_lemmatizer (WordNetLemmatizer): Used for lemmatizing concepts.
        concept_cache_store (dict): Caches extracted concepts for efficiency.
        nlp_tool (spacy.Language): SpaCy NLP model for text processing.
        similarity_threshold_value (float): Threshold for adding edges based on similarity.
    """

    def __init__(self):
        """Initializes the KnowledgeGraph with essential components."""
        self.graph_structure = nx.Graph()
        self.word_lemmatizer = WordNetLemmatizer()
        self.concept_cache_store = {}
        self.nlp_tool = self._load_spacy_model()
        self.similarity_threshold_value = 0.8

    def build_graph(self, doc_chunks: List[Any], llm_tool: Any, embedding_tool: Any):
        """Builds the knowledge graph by adding nodes, creating embeddings, extracting concepts, and adding edges.

        Args:
            doc_chunks (list): List of document chunks.
            llm_tool: Large language model instance.
            embedding_tool: Embedding model instance.
        """
        self.add_nodes(doc_chunks)
        embeddings = self._create_embeddings(doc_chunks, embedding_tool)
        self._extract_concepts(doc_chunks, llm_tool)
        self._add_edges(embeddings)

    def add_nodes(self, doc_chunks: List[Any]):
        """Adds nodes to the graph from document chunks.

        Args:
            doc_chunks (list): List of document chunks to add as nodes.
        """
        for index, chunk in enumerate(doc_chunks):
            self.graph_structure.add_node(index, content=chunk.page_content)

    def _create_embeddings(self, document_chunks: List[Any], embedding_model: Any):
        """Creates embeddings for the document chunks using the embedding model.

        Args:
            document_chunks (list): List of document chunks.
            embedding_model: Embedding model instance.

        Returns:
            numpy.ndarray: Array of embeddings for the document chunks.
        """
        texts = [chunk.page_content for chunk in document_chunks]
        return embedding_model.embed_documents(texts)

    def compute_similarity_matrix(self, embeddings: Any):
        """Computes the cosine similarity matrix for given embeddings.

        Args:
            embeddings (numpy.ndarray): Array of embeddings.

        Returns:
            numpy.ndarray: Cosine similarity matrix.
        """
        return cosine_similarity(embeddings)

    def _load_spacy_model(self) -> spacy.Language:
        """Loads the SpaCy NLP model, downloading it if not available.

        Returns:
            spacy.Language: Loaded SpaCy NLP model.
        """
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading SpaCy model...")
            download("en_core_web_sm")
            return spacy.load("en_core_web_sm")

    def _extract_concepts_and_entities(self, content: str, llm: Any) -> List[str]:
        """Extracts concepts and named entities from the content using spaCy and a large language model.

        Args:
            content (str): The content from which to extract concepts and entities.
            llm: An instance of a large language model.

        Returns:
            list: A list of extracted concepts and entities.
        """
        if content in self.concept_cache_store:
            return self.concept_cache_store[content]

        # Extract named entities using spaCy
        doc = self.nlp_tool(content)
        named_entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "WORK_OF_ART"]]

        # Extract general concepts using LLM
        concept_extraction_prompt = PromptTemplate(
            input_variables=["text"],
            template="Extract key concepts (excluding named entities) from the following text:\n\n{text}\n\nKey concepts:"
        )
        concept_chain = concept_extraction_prompt | llm.with_structured_output(Concepts)
        general_concepts = concept_chain.invoke({"text": content}).concepts_list

        # Combine named entities and general concepts
        all_concepts = list(set(named_entities + general_concepts))

        self.concept_cache_store[content] = all_concepts
        return all_concepts

    def _extract_concepts(self, document_chunks: List[Any], llm: Any):
        """Extracts concepts for all document chunks using multi-threading.

        Args:
            document_chunks (list): List of document chunks.
            llm: Large language model instance.
        """
        with ThreadPoolExecutor() as executor:
            future_to_node = {executor.submit(self._extract_concepts_and_entities, chunk.page_content, llm): i
                              for i, chunk in enumerate(document_chunks)}

            for future in tqdm(as_completed(future_to_node), total=len(document_chunks),
                               desc="Extracting concepts and entities"):
                node = future_to_node[future]
                concepts = future.result()
                self.graph_structure.nodes[node]['concepts'] = concepts

    def _add_edges(self, embeddings: Any):
        """Adds edges to the graph based on the similarity of embeddings and shared concepts.

        Args:
            embeddings (numpy.ndarray): Array of embeddings for the document chunks.
        """
        similarity_matrix = self.compute_similarity_matrix(embeddings)
        num_nodes = len(self.graph_structure.nodes)

        for node1 in tqdm(range(num_nodes), desc="Adding edges"):
            for node2 in range(node1 + 1, num_nodes):
                similarity_score = similarity_matrix[node1][node2]
                if similarity_score > self.similarity_threshold_value:
                    shared_concepts = set(self.graph_structure.nodes[node1]['concepts']) & set(
                        self.graph_structure.nodes[node2]['concepts'])
                    edge_weight = self.calculate_edge_weight(node1, node2, similarity_score, shared_concepts)
                    self.graph_structure.add_edge(node1, node2, weight=edge_weight,
                                        similarity=similarity_score,
                                        shared_concepts=list(shared_concepts))

    def calculate_edge_weight(self, node1: int, node2: int, similarity_score: float, shared_concepts: set, alpha: float = 0.7, beta: float = 0.3) -> float:
        """Calculates the weight of an edge based on similarity and shared concepts.

        Args:
            node1 (int): First node index.
            node2 (int): Second node index.
            similarity_score (float): Cosine similarity score between nodes.
            shared_concepts (set): Shared concepts between nodes.
            alpha (float): Weight for similarity score. Default is 0.7.
            beta (float): Weight for shared concepts. Default is 0.3.

        Returns:
            float: Calculated edge weight.
        """
        max_shared_concepts = min(len(self.graph_structure.nodes[node1]['concepts']), len(self.graph_structure.nodes[node2]['concepts']))
        normalized_shared_concepts = len(shared_concepts) / max_shared_concepts if max_shared_concepts > 0 else 0
        return alpha * similarity_score + beta * normalized_shared_concepts

    def lemmatize_concept(self, concept: str) -> str:
        """Lemmatizes a given concept.

        Args:
            concept (str): Concept to lemmatize.

        Returns:
            str: Lemmatized concept.
        """
        return ' '.join([self.word_lemmatizer.lemmatize(word) for word in concept.lower().split()])