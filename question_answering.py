# question_answering.py
import streamlit as st
import openai
import pickle
from rank_bm25 import BM25Okapi

import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Set up OpenAI API key
openai.api_key = st.secrets["openai"]["api_key"]

class BM25Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        tokenized_chunks = [chunk['content'].split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

    def retrieve(self, query, top_k=5):
        tokenized_query = query.split()
        top_chunks = self.bm25.get_top_n(tokenized_query, self.chunks, n=top_k)
        return top_chunks

# GPT-3.5 Generator Model for generating answers
class GPT35AnswerGenerator:
    def __init__(self):
        pass

    def generate_answer(self, relevant_chunks, query, max_tokens=200):
        prompt = f"Answer the question: {query}\nBased on the following legal information:\n{relevant_chunks}"

        # Call the OpenAI API using 'openai.completions.create()' method
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # or 'gpt-4' if you have access
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on legal information."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200  # Limit the length of the response
        )

        # Return the generated answer text
        return response.choices[0].message.content

# Combine the Retriever and Generator for question answering
class RetrieverGeneratorQA:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def answer(self, query, max_length=200):
        relevant_chunks = self.retriever.retrieve(query)
        context = " ".join([chunk['content'] for chunk in relevant_chunks])
        answer = self.generator.generate_answer(context, query, max_tokens=max_length)
        sources = [{'source': chunk['source'], 'text': chunk['content']} for chunk in relevant_chunks]
        return answer, sources

# Load preprocessed BM25 index and chunks from a file
def load_preprocessed_data(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

# Streamlit UI for question answering
def main():
    st.title("Document Question Answering System")

    save_file = 'bm25_model.pkl'
    retriever = load_preprocessed_data(save_file)

    generator = GPT35AnswerGenerator()
    qa_system = RetrieverGeneratorQA(retriever, generator)

    question = st.text_input("Ask a question about the documents:")

    if st.button("Get Answer"):
        if question:
            answer, sources = qa_system.answer(question)
            st.subheader("Answer:")
            st.write(answer)

            st.subheader("Sources:")
            for source in sources:
                st.write(f"**Document:** {source['source']}")
                st.write(f"**Excerpt:** {source['text']}")
                st.write("---")
        else:
            st.write("Please enter a question.")

if __name__ == "__main__":
    main()
