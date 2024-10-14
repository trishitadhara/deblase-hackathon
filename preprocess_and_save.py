import os
import pickle
from rank_bm25 import BM25Okapi
import PyPDF2

# Function to extract text from PDFs
def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return text

# Function to load all files from the directory (handling both PDFs and text files)
def load_files_from_directory(directory_path):
    documents = []
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        if os.path.isfile(file_path):
            if file_name.lower().endswith('.pdf'):
                text = extract_text_from_pdf(file_path)
                if text:
                    documents.append({'name': file_name, 'content': text})
            else:
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        documents.append({'name': file_name, 'content': file.read()})
                except UnicodeDecodeError:
                    with open(file_path, 'r', encoding='ISO-8859-1') as file:
                        documents.append({'name': file_name, 'content': file.read()})
    return documents

# Function to chunk text into smaller pieces and track document names
def chunk_text_with_source(documents, chunk_size=1000):
    chunks = []
    for doc in documents:
        text = doc['content']
        doc_name = doc['name']
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            chunks.append({'content': chunk, 'source': doc_name})
    return chunks

# BM25 Retriever Model for finding relevant text chunks
class BM25Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        tokenized_chunks = [chunk['content'].split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

    def retrieve(self, query, top_k=5):
        tokenized_query = query.split()
        top_chunks = self.bm25.get_top_n(tokenized_query, self.chunks, n=top_k)
        return top_chunks

# Save BM25 index and chunked data to a file
def save_preprocessed_data(file_name, retriever):
    with open(file_name, 'wb') as f:
        pickle.dump(retriever, f)

# Main function to process and save the preprocessed data
def main(directory_path, save_file='bm25_model.pkl'):
    documents = load_files_from_directory(directory_path)
    chunks = chunk_text_with_source(documents)
    retriever = BM25Retriever(chunks)
    save_preprocessed_data(save_file, retriever)
    print(f"Preprocessing complete. BM25 model saved to {save_file}")


# Example usage
if __name__ == "__main__":
    directory_path = './docs'  # Update with your directory path
    save_file = 'bm25_model.pkl'  # The path where the pkl file will be saved

    main(directory_path, save_file)
