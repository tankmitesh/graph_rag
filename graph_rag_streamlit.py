import streamlit as st
from graph_rag import GraphRAG
from langchain_community.document_loaders import WebBaseLoader

def main():

    st.title("Graph RAG")

    # Input for web link
    user_web_link = st.sidebar.text_input("Paste a web link to fetch information:")

    if user_web_link:
        try:
            # Load the web page content from the link
            st.write("Fetching content from the provided link...")
            web_content_loader = WebBaseLoader(user_web_link)
            loaded_documents = web_content_loader.load()

            # Initialize GraphRAG
            graph_rag_instance = GraphRAG(loaded_documents)

            # Process the documents and create the graph
            graph_rag_instance.process_documents(loaded_documents)

            # Input for user query
            user_query = st.text_input("Enter your query:")

            if user_query:
                st.write("Processing your query...")
                query_response = graph_rag_instance.query(user_query)
                st.write("Response:", query_response['content'])

        except Exception as error:
            st.error(f"An error occurred: {error}")

if __name__ == "__main__":
    main()