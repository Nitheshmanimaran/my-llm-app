import streamlit as st
import requests
import os
import time
from requests.auth import HTTPBasicAuth
from pinecone import Pinecone
import hashlib
from pymongo import MongoClient
import mimetypes
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime
import tempfile

# Get configurations from Streamlit secrets
MONGO_URI = st.secrets["MONGO_URI"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]
INDEX_NAME = st.secrets["INDEX_NAME"]
EMBEDDING_API_URL = st.secrets["EMBEDDING_API_URL"]
#SHARED_DIR = st.secrets["SHARED_DIR"]

# Create a temporary directory for file storage
SHARED_DIR = os.path.join(tempfile.gettempdir(), 'streamlit_uploads')
os.makedirs(SHARED_DIR, exist_ok=True)

client = MongoClient(MONGO_URI)
db = client[INDEX_NAME]

# Initialize Pinecone
pc = Pinecone(
    api_key=PINECONE_API_KEY,
    ssl_verify=False
)
index = pc.Index(INDEX_NAME)

def get_embeddings(text_chunks):
    """Get embeddings from the API"""
    try:
        print(f"Sending {len(text_chunks)} chunks to embedding API...")
        response = requests.post(
            EMBEDDING_API_URL,
            json={"text_chunks": text_chunks},
            headers={"Content-Type": "application/json"},
            verify=False
        )
        response.raise_for_status()
        embeddings = response.json()['embeddings']
        print(f"Successfully received embeddings. First embedding shape: {len(embeddings[0])} dimensions")
        return embeddings
    except Exception as e:
        st.error(f"Error getting embeddings: {str(e)}")
        return None

def get_query_embedding(query_text):
    """Get embedding for the query text"""
    try:
        response = requests.post(
            EMBEDDING_API_URL,
            json={"text_chunks": [query_text]},
            headers={"Content-Type": "application/json"},
            verify=False
        )
        response.raise_for_status()
        return response.json()['embeddings'][0]
    except Exception as e:
        st.error(f"Error getting query embedding: {str(e)}")
        return None

def search_similar_documents(query_embedding, top_k=5):
    """Search similar documents in Pinecone"""
    try:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return results
    except Exception as e:
        st.error(f"Error searching documents: {str(e)}")
        return None

def upload_section():
    """Document Upload Section"""
    st.title("Document Upload")
    
    # Ensure the shared directory exists
    #os.makedirs(SHARED_DIR, exist_ok=True)

    # File uploader
    uploaded_files = st.file_uploader("Upload your documents", type=["pdf", "txt", "doc", "docx", "jpg", "png"], accept_multiple_files=True)

    # Display uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.write(f"Uploaded file: {uploaded_file.name}")
            # Save the file to the shared directory
            file_path = os.path.join(SHARED_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success("Files uploaded successfully!")

        # Trigger Airflow DAG
        response = requests.post(
            "http://localhost:1128/api/v1/dags/data_ingestion_pipeline/dagRuns",
            json={"conf": {"file_path": SHARED_DIR}},
            auth=HTTPBasicAuth('admin', 'password')
        )

        if response.status_code == 200:
            dag_run_id = response.json()['dag_run_id']
            st.success(f"Airflow DAG triggered successfully! DAG Run ID: {dag_run_id}")

            # Poll for DAG completion
            dag_status = "running"
            while dag_status == "running":
                time.sleep(10)  # Wait for 10 seconds before polling again
                status_response = requests.get(
                    f"http://localhost:1128/api/v1/dags/data_ingestion_pipeline/dagRuns/{dag_run_id}",
                    auth=HTTPBasicAuth('admin', 'password')
                )
                dag_status = status_response.json()['state']
                st.write(f"DAG Status: {dag_status}")

            if dag_status == "success":
                st.success("DAG completed successfully!")
            else:
                st.error("DAG failed or was not completed successfully.")
        else:
            st.error("Failed to trigger Airflow DAG.")

def extract_text_from_file(file_content, file_type):
    """Extract text from file based on its type"""
    try:
        if file_type == 'application/pdf':
            # For PDF files
            from io import BytesIO
            from PyPDF2 import PdfReader
            pdf = PdfReader(BytesIO(file_content))
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""  # Handle empty pages
            return text.strip()
            
        elif 'text' in file_type:
            # For text files
            try:
                return file_content.decode('utf-8')
            except UnicodeDecodeError:
                return file_content.decode('latin-1')  # Fallback encoding
                
        elif 'document' in file_type:
            # For Word documents
            import docx2txt
            from io import BytesIO
            return docx2txt.process(BytesIO(file_content))
            
        else:
            st.warning(f"Unsupported file type: {file_type}")
            return None
            
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return None

def search_section():
    """Document Search and Comparison Section"""
    st.title("Document Search & Comparison")
    
    # Document upload for comparison
    uploaded_file = st.file_uploader("Upload a document to compare", type=["pdf", "txt", "doc", "docx"])
    
    # Query input (we'll use this later)
    query = st.text_input("Enter your search query (optional)")
    
    if uploaded_file:
        # Create a temporary file path
        temp_file_path = os.path.join(SHARED_DIR, uploaded_file.name)
        
        # Read file content
        file_content = uploaded_file.read()
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        # Check if file exists in processed_files
        existing_doc = db['processed_files'].find_one({'hash': file_hash})
        
        if existing_doc:
            st.info(f"Document '{uploaded_file.name}' found in system")
            
            # Display basic metadata
            st.json({
                "filename": existing_doc.get('filename'),
                "processed_at": existing_doc.get('processed_at'),
                "total_chunks": existing_doc.get('total_chunks'),
                "hash": file_hash
            })
            
            # Retrieve embeddings from Pinecone using hash-based IDs
            try:
                total_chunks = existing_doc.get('total_chunks', 0)
                vector_ids = [f"{file_hash}_{i}" for i in range(total_chunks)]
                
                # Fetch vectors from Pinecone
                fetch_response = index.fetch(ids=vector_ids)
                
                if fetch_response.vectors:
                    st.success(f"Retrieved {len(fetch_response.vectors)} embeddings from Pinecone")
                    
                    # Store embeddings for later use
                    retrieved_embeddings = []
                    for vid in vector_ids:
                        if vid in fetch_response.vectors:
                            vector_data = fetch_response.vectors[vid]
                            retrieved_embeddings.append({
                                'id': vid,
                                'embedding': vector_data.values,
                                'metadata': vector_data.metadata
                            })
                    
                    # Display some metadata about retrieved vectors
                    st.subheader("Retrieved Vector Information")
                    for i, vec in enumerate(retrieved_embeddings[:3]):  # Show first 3 for brevity
                        with st.expander(f"Vector {vec['id']}"):
                            st.json({
                                'id': vec['id'],
                                'embedding_dimension': len(vec['embedding']),
                                'metadata': vec['metadata']
                            })
                    
                    if len(retrieved_embeddings) > 3:
                        st.info(f"... and {len(retrieved_embeddings) - 3} more vectors")
                    
                    # Store embeddings in session state for later use
                    st.session_state['retrieved_embeddings'] = retrieved_embeddings
                    
                else:
                    st.error("No vectors found in Pinecone")
                
            except Exception as e:
                st.error(f"Error retrieving vectors from Pinecone: {str(e)}")
            
        else:
            st.warning(f"New document detected. Processing required.")
            
            if st.button("Process Document"):
                try:
                    # Save file temporarily
                    with open(temp_file_path, "wb") as f:
                        f.write(file_content)
                    
                    # Get file type
                    file_type, _ = mimetypes.guess_type(uploaded_file.name)
                    
                    # Extract text
                    text = extract_text_from_file(file_content, file_type)
                    
                    if text:
                        # Create chunks
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=2000,
                            chunk_overlap=200
                        )
                        chunks = text_splitter.split_text(text)
                        
                        if chunks:
                            # Get embeddings
                            embeddings = get_embeddings(chunks)
                            
                            if embeddings and len(embeddings) == len(chunks):
                                # Store in MongoDB without file path
                                collection_name = 'pdfs' if file_type == 'application/pdf' else 'docs' if 'document' in file_type else 'texts'
                                collection = db[collection_name]
                                
                                # Store in MongoDB without permanent file path
                                doc = {
                                    'filename': uploaded_file.name,
                                    'hash': file_hash,
                                    'chunks': chunks,
                                    'processed_at': datetime.now()
                                }
                                collection.insert_one(doc)
                                
                                # Store in Pinecone
                                vectors_to_upsert = []
                                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                                    vector_id = f"{file_hash}_{i}"
                                    metadata = {
                                        'file_name': uploaded_file.name,
                                        'file_hash': file_hash,
                                        'chunk_id': i,
                                        'chunk_text': chunk,
                                        'file_type': file_type,
                                        'processed_at': datetime.now().isoformat()
                                    }
                                    vectors_to_upsert.append((vector_id, embedding, metadata))
                                
                                # Upsert to Pinecone
                                batch_size = 100
                                for i in range(0, len(vectors_to_upsert), batch_size):
                                    batch = vectors_to_upsert[i:i + batch_size]
                                    index.upsert(vectors=batch)
                                
                                # Record in processed_files
                                db['processed_files'].insert_one({
                                    'hash': file_hash,
                                    'filename': uploaded_file.name,
                                    'processed_at': datetime.now(),
                                    'total_chunks': len(chunks)
                                })
                                
                                st.success(f"Successfully processed document")
                                
                                # Store embeddings in session state
                                st.session_state['document_embeddings'] = [
                                    {
                                        'id': vid,
                                        'embedding': emb,
                                        'metadata': meta
                                    } for vid, emb, meta in vectors_to_upsert
                                ]
                            
                            else:
                                st.error("Failed to generate embeddings")
                        else:
                            st.error("No text chunks created")
                    else:
                        st.error("Failed to extract text from document")
                        
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)

    # Query handling will be added in the next phase
    if query:
        st.info("Query processing will be implemented in the next phase")

def main():
    st.set_page_config(page_title="Document Management System", layout="wide")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📤 Upload Documents", "🔍 Search Documents"])
    
    with tab1:
        upload_section()
    
    with tab2:
        search_section()

# App title
#st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")

# Replicate Credentials
    with st.sidebar:
        st.title('🦙💬 Llama 2 Chatbot')
        if 'REPLICATE_API_TOKEN' in st.secrets:
            st.success('API key already provided!', icon='✅')
            replicate_api = st.secrets['REPLICATE_API_TOKEN']
        else:
            replicate_api = st.text_input('Enter Replicate API token:', type='password')
            if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
                st.warning('Please enter your credentials!', icon='⚠️')
            else:
                st.success('Proceed to entering your prompt message!', icon='👉')
        os.environ['REPLICATE_API_TOKEN'] = replicate_api
    
        st.subheader('Models and parameters')
        selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama2-7B', 'Llama2-13B'], key='selected_model')
        if selected_model == 'Llama2-7B':
            llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
        elif selected_model == 'Llama2-13B':
            llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
        temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
        top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
        max_length = st.sidebar.slider('max_length', min_value=32, max_value=128, value=120, step=8)
        st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')
    
    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    
    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
    
    # Function for generating LLaMA2 response. Refactored from https://github.com/a16z-infra/llama2-chatbot
    def generate_llama2_response(prompt_input):
        string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
        for dict_message in st.session_state.messages:
            if dict_message["role"] == "user":
                string_dialogue += "User: " + dict_message["content"] + "\n\n"
            else:
                string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
        output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', 
                               input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                                      "temperature":temperature, "top_p":top_p, "max_length":max_length, "repetition_penalty":1})
        return output
    
    # User-provided prompt
    if prompt := st.chat_input(disabled=not replicate_api):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
    
    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_llama2_response(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)

if __name__ == "__main__":
    main()
