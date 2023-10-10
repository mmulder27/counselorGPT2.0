

from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import openai
##import tiktoken
import pinecone
import streamlit as st
import time

#NOTE: CounselorGPT2.0 is no longer operational.

#Initialize OpenAIEmbeddings and Pinecone
embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
pinecone.init(api_key=st.secrets["PINECONE_API_KEY"],environment=st.secrets["PINECONE_API_ENV"])
index_name = "counselorgpt2index"
index = pinecone.Index(index_name)

#Initialize retriever for hybrid search of Pinecone index
bm25_encoder = BM25Encoder().load("bm25_values.json")
retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder, index=index)


#Create prompt template
prompt_template = """You are an academic counselor at UCLA who has been provided access to a database of course descriptions, major descriptions, institutional policies, and student course reviews. If you are asked about student reviews of a course, please quote students in your response. If you do not know the answer, direct them to bruinwalk.com for student-review related questions and to catalog.registrar.ucla.edu for all other questions. Answer the user's questions based on this content:

{context}

Question: {question}"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

#Create conversation chain
chain = load_qa_chain((OpenAI(temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])), chain_type="stuff", prompt=PROMPT)


#GUI implementation
st.set_page_config(page_title="CounselorGPT 2.0", page_icon=":robot:")
st.header("CounselorGPT 2.0")
st.write("*CounselorGPT 2.0 is no longer functional due to 1) the rapid change in langchain documentation, rendering CounselorGPT 2.0's code obsolete every few weeks and 2) the price of maintaining its knowledge database.*")    
st.write("**Having read through UCLA's registrar and Bruinwalk's course reviews, I'm here to provide comprehensive answers to all your academic inquiries about UCLA.**")
with st.expander("Examples of questions you might ask:"):
    st.write("*What classes can I take to learn about Roman architecture?*")
    st.write("*What are some examples of courses I can take to satisfy my diversity requirement for the College of Letters and Sciences?*")
    st.write("*What have students said about PHYSCI 128 taught by Professor Hsiao?*")

#Check for "messages" attribute in session state and initialize it if necessary
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept user input
if prompt := st.chat_input("How can I assist you?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        
        try:
            message_placeholder = st.empty()
            full_response = ""
            docs = retriever.get_relevant_documents(prompt)
            assistant_response = chain({"input_documents": docs, "question": prompt}, return_only_outputs=True)['output_text'].replace("\n","")

            # Simulate stream of response with milliseconds delay
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})  
        except Exception:
            message_placeholder.markdown("Please elaborate.")
            st.session_state.messages.append({"role": "assistant", "content": "Please elaborate."}) 
    
    
    
