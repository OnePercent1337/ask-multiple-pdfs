import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import openai

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def handle_userinput(user_question):
    # Hardcoded responses for specific queries
    if user_question.lower() == "who are you?":
        assistant_reply = "I am a chatbot created by ChainHub.io to assist with crypto-related questions."
    else:
        # Existing logic here...
        conversation = [
            {"system": "You are a specialized chatbot created by ChainHub.io. You should only refer to yourself as being developed and maintained by ChainHub.io. Your purpose is to assist with crypto-related queries, and you should only generate responses based on the provided documents. You are not a general-purpose assistant and should refrain from saying you were made by OpenAI."},
            {"role": "user", "content": user_question}
        ]
        
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=conversation,
            max_tokens=8000
        )
        assistant_reply = response.choices[0].text.strip()


    # Update the chat history session state
    if st.session_state.chat_history is None:
        st.session_state.chat_history = []

    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})

    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.write(user_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
            

     
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize session state if it doesn't exist
    if 'conversation' not in st.session_state:
        st.session_state['conversation'] = None
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'pdf_uploaded' not in st.session_state:
        st.session_state['pdf_uploaded'] = False  # NEW: Track PDF upload status

    # Display welcome message if the chat history is empty
    if len(st.session_state.chat_history) == 0:
        welcome_message = "Hello! Welcome to Chainhub.io's Chatbot. I'm here to help you with all kinds of crypto-related questions. Feel free to ask!"
        st.write(bot_template.replace("{{MSG}}", welcome_message), unsafe_allow_html=True)

    st.header("Chat with multiple PDFs :books:")

    # Input fields for bot name and system context
    bot_name = st.text_input("Name the Bot:")
    system_context = st.text_area("Give the bot system context:")

    # NEW CODE HERE: Display PDF upload option only if not uploaded
    if not st.session_state['pdf_uploaded']:
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.session_state['pdf_uploaded'] = True  # NEW: Mark PDF as uploaded
    else:
        # Allow chat interaction only if PDF has been uploaded
        user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            handle_userinput(user_question)

if __name__ == '__main__':
    main()