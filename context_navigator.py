from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.callbacks.manager import get_openai_callback
import heapq
from answer_check import AnswerCheck
from typing import List, Tuple, Dict, Any


class QueryEngine:
    """Handles querying the knowledge graph and retrieving relevant answers."""

    def __init__(self, vector_storage, knowledge_graph_tool, language_model_tool):
        """Initializes the QueryEngine with required components.

        Args:
            vector_storage: Vector store for document retrieval.
            knowledge_graph_tool: Instance of the KnowledgeGraph class.
            language_model_tool: Large language model instance.
        """
        self.vector_storage = vector_storage
        self.knowledge_graph_tool = knowledge_graph_tool
        self.language_model_tool = language_model_tool
        self.max_context_length = 4000
        self.answer_validation_chain = self._create_answer_validation_chain()

    def _create_answer_validation_chain(self):
        """Creates a chain to check if the context provides a complete answer to the query.

        Returns:
            Chain: A chain to check if the context provides a complete answer.
        """
        validation_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template=(
                "Given the query: '{query}'\n\nAnd the current context:\n{context}\n\n"
                "Does this context provide a complete answer to the query? If yes, provide the answer. "
                "If no, state that the answer is incomplete.\n\nIs complete answer (Yes/No):\nAnswer (if complete):"
            ),
        )
        return validation_prompt | self.language_model_tool.with_structured_output(AnswerCheck)

    def _check_answer(self, user_query: str, current_context: str) -> Tuple[bool, str]:
        """Checks if the current context provides a complete answer to the query.

        Args:
            user_query (str): The query to be answered.
            current_context (str): The current context.

        Returns:
            Tuple[bool, str]: A tuple containing:
                - is_complete (bool): Whether the context provides a complete answer.
                - answer (str): The answer based on the context, if complete.
        """
        response = self.answer_validation_chain.invoke({"query": user_query, "context": current_context})
        return response.is_complete, response.answer

    def _expand_context(
        self, query: str, relevant_documents: List[Any]
    ) -> Tuple[str, List[int], Dict[int, str], str]:
        """Expands the context by traversing the knowledge graph using a Dijkstra-like approach.

        Args:
            query (str): The query to be answered.
            relevant_documents (List[Any]): A list of relevant documents to start the traversal.

        Returns:
            Tuple[str, List[int], Dict[int, str], str]: A tuple containing:
                - expanded_context (str): The accumulated context from traversed nodes.
                - traversal_path (List[int]): The sequence of node indices visited.
                - filtered_content (Dict[int, str]): A mapping of node indices to their content.
                - final_answer (str): The final answer found, if any.
        """
        expanded_context = ""
        traversal_path = []
        visited_concepts = set()
        filtered_content = {}
        final_answer = ""

        priority_queue = []
        distances = {}

        print("\nTraversing the knowledge graph:")

        for document in relevant_documents:
            closest_nodes = self.vector_storage.similarity_search_with_score(
                document.page_content, k=1
            )
            closest_node_content, similarity_score = closest_nodes[0]

            closest_node = next(
                node
                for node in self.knowledge_graph_tool.graph.nodes
                if self.knowledge_graph_tool.graph.nodes[node]["content"]
                == closest_node_content.page_content
            )

            priority = 1 / similarity_score
            heapq.heappush(priority_queue, (priority, closest_node))
            distances[closest_node] = priority

        while priority_queue:
            current_priority, current_node = heapq.heappop(priority_queue)

            if current_priority > distances.get(current_node, float("inf")):
                continue

            if current_node not in traversal_path:
                traversal_path.append(current_node)
                node_content = self.knowledge_graph_tool.graph.nodes[current_node]["content"]
                node_concepts = self.knowledge_graph_tool.graph.nodes[current_node]["concepts"]

                filtered_content[current_node] = node_content
                expanded_context += "\n" + node_content if expanded_context else node_content

                is_complete, answer = self._check_answer(query, expanded_context)
                if is_complete:
                    final_answer = answer
                    break

                node_concepts_set = set(
                    self.knowledge_graph_tool.lemmatize_concept(concept)
                    for concept in node_concepts
                )
                if not node_concepts_set.issubset(visited_concepts):
                    visited_concepts.update(node_concepts_set)

                    for neighbor in self.knowledge_graph_tool.graph.neighbors(current_node):
                        edge_data = self.knowledge_graph_tool.graph[current_node][neighbor]
                        edge_weight = edge_data["weight"]

                        distance = current_priority + (1 / edge_weight)

                        if distance < distances.get(neighbor, float("inf")):
                            distances[neighbor] = distance
                            heapq.heappush(priority_queue, (distance, neighbor))

        if not final_answer:
            print("\nGenerating final answer...")
            response_prompt = PromptTemplate(
                input_variables=["query", "context"],
                template=(
                    "Based on the following context, please answer the query.\n\n"
                    "Context: {context}\n\nQuery: {query}\n\nAnswer:"
                ),
            )
            response_chain = response_prompt | self.language_model_tool
            input_data = {"query": query, "context": expanded_context}
            final_answer = response_chain.invoke(input_data)

        return expanded_context, traversal_path, filtered_content, final_answer

    def query(self, query: str) -> Tuple[str, List[int], Dict[int, str]]:
        """Processes a query by retrieving relevant documents, expanding the context, and generating the final answer.

        Args:
            query (str): The query to be answered.

        Returns:
            Tuple[str, List[int], Dict[int, str]]: A tuple containing:
                - final_answer (str): The final answer to the query.
                - traversal_path (List[int]): The traversal path of nodes in the knowledge graph.
                - filtered_content (Dict[int, str]): The filtered content of nodes.
        """
        with get_openai_callback() as callback:
            print(f"\nProcessing query: {query}")
            relevant_documents = self._retrieve_relevant_documents(query)
            expanded_context, traversal_path, filtered_content, final_answer = self._expand_context(
                query, relevant_documents
            )

            if not final_answer:
                print("\nGenerating final answer...")
                response_prompt = PromptTemplate(
                    input_variables=["query", "context"],
                    template=(
                        "Based on the following context, please answer the query.\n\n"
                        "Context: {context}\n\nQuery: {query}\n\nAnswer:"
                    ),
                )
                response_chain = response_prompt | self.language_model_tool
                input_data = {"query": query, "context": expanded_context}
                final_answer = response_chain.invoke(input_data)

            print(f"\nFinal Answer: {final_answer}")
            print(f"\nTotal Tokens: {callback.total_tokens}")
            print(f"Prompt Tokens: {callback.prompt_tokens}")
            print(f"Completion Tokens: {callback.completion_tokens}")
            print(f"Total Cost (USD): ${callback.total_cost}")

        return final_answer, traversal_path, filtered_content

    def _retrieve_relevant_documents(self, query: str) -> List[Any]:
        """Retrieves relevant documents based on the query using the vector store.

        Args:
            query (str): The query to be answered.

        Returns:
            List[Any]: A list of relevant documents.
        """
        print("\nRetrieving relevant documents...")
        retriever = self.vector_storage.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )
        compressor = LLMChainExtractor.from_llm(self.language_model_tool)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )
        return compression_retriever.invoke(query)
