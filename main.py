import os
import gradio as gr
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.chains import RetrievalQA, create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory.chat_message_histories import ChatMessageHistory
from langchain.callbacks import LangChainTracer
from langchain.callbacks.manager import CallbackManager
from langchain_community.tools import DuckDuckGoSearchRun
import warnings
from dotenv import load_dotenv
warnings.filterwarnings("ignore")
load_dotenv()

class CricketExpertBot:
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None
        self.chat_history = ChatMessageHistory()
        self.web_search_tool = DuckDuckGoSearchRun()
        self.awaiting_internet_search_confirmation = False
        self.last_unanswered_query = None
        self.setup_components()
        
    def setup_components(self):
        print("Setting up Cricket Bot Components")

        self.embeddings = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBEDDINGS_MODEL"))

        self.llm = Ollama(
            model=os.getenv("OLLAMA_LLM_MODEL"),
            temperature=0.7,
            callback_manager=CallbackManager([LangChainTracer()])
        )
        
        self.load_pdf_data()
        
        self.setup_qa_chain()
        
    def load_pdf_data(self):
        try:
            print("Loading PDF Data")

            loader = PyPDFLoader(os.getenv("PDF_FILE_PATH"))
            documents = loader.load()
            print(f"Loaded {len(documents)} pages from PDF")
            
            # Split text
            text_splitter = CharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                separator="\n"
            )
            texts = text_splitter.split_documents(documents)
            print(f"Split into {len(texts)} chunks")
            
            # Create vector store
            self.vector_store = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                collection_name="cricket_players",
                persist_directory="./chroma_db"
            )
            print("Vector store created successfully")
            
        except Exception as e:
            print(f"Error loading PDF: {e}")
            self.vector_store = Chroma(
                embedding_function=self.embeddings,
                collection_name="cricket_players",
                persist_directory="./chroma_db"
            )

    def setup_qa_chain(self):
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "just reformulate it if needed and otherwise return it as is."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, prompt
        )

        qa_system_prompt = """
            You are CricketBot, an expert cricket assistant with deep knowledge about cricket players, statistics, and history.
            Use the following retrieved context to answer the question. If you don't know the answer based on the context, 
            say so honestly and ask the user if they would like you to search the internet for the answer. Be conversational and helpful.

            Context: {context}
            """
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        self.document_combiner_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        self.qa_chain = create_retrieval_chain(history_aware_retriever, self.document_combiner_chain)
        print("QA Chain setup complete")

    def _is_positive_response(self, text):
        positive_keywords = ["yes", "yeah", "yep", "sure", "ok", "please", "go ahead", "search", "y"]
        return any(keyword in text.lower() for keyword in positive_keywords)

    def process_user_input(self, user_question):
        if self.awaiting_internet_search_confirmation:
            if self._is_positive_response(user_question):
                self.awaiting_internet_search_confirmation = False
                query_to_search = self.last_unanswered_query if self.last_unanswered_query else user_question
                self.last_unanswered_query = None

                self.chat_history.add_user_message(user_question)

                web_answer = self.web_search_and_respond(query_to_search)
                return web_answer
            else:
                self.awaiting_internet_search_confirmation = False
                self.last_unanswered_query = None
                response = "Okay, I won't search the internet for that. Is there anything else I can help you with from my current knowledge?"
                self.chat_history.add_user_message(user_question)
                self.chat_history.add_ai_message(response)
                return response
        else:
            try:
                response = self.qa_chain.invoke(
                    {
                        "chat_history": self.chat_history.messages, 
                        "input": user_question
                    }
                )
                
                answer = response.get("answer", "")
                lower_answer = answer.lower()
                no_ans_from_llm_words = [
                    "i don't know",
                    "not based on the context",
                    "not able to find",
                    "isn't mentioned",
                    "only have information about"
                ]
                check_at_least_one_word_lambda = lambda sentence, word_list: any(word.lower() in sentence.lower() for word in word_list)
                if check_at_least_one_word_lambda(lower_answer,no_ans_from_llm_words):
                    
                    self.awaiting_internet_search_confirmation = True
                    self.last_unanswered_query = user_question
                    follow_up_message = f"{answer} Would you like me to search the internet for '{user_question}'?"
                    
                    
                    self.chat_history.add_user_message(user_question)
                    self.chat_history.add_ai_message(follow_up_message)
                    return follow_up_message
                else:
                    
                    self.chat_history.add_user_message(user_question)
                    self.chat_history.add_ai_message(answer)
                    return answer
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                self.chat_history.add_user_message(user_question)
                self.chat_history.add_ai_message(error_msg)
                return error_msg

    def web_search_and_respond(self, query):
        
        try:
            print(f"Performing web search for: {query}")
            search_results = self.web_search_tool.run(query)
            
            web_search_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are CricketBot. Based on the following web search results, answer the user's question concisely. If the search results don't contain a clear answer, state that."),
                ("human", f"User question: {query}\n\nWeb Search Results:\n{search_results}"),
            ])
            
            summarized_answer = self.llm.invoke(web_search_prompt.format(query=query, search_results=search_results))
            
            # The user's confirmation already added their message. Now add AI's response.
            self.chat_history.add_ai_message(summarized_answer)
            return summarized_answer
        except Exception as e:
            error_msg = f"Sorry, I couldn't perform the web search: {str(e)}"
            self.chat_history.add_ai_message(error_msg)
            return error_msg
            
    def clear_history(self):
        self.chat_history.clear()
        self.awaiting_internet_search_confirmation = False
        self.last_unanswered_query = None
        return "Chat history cleared"
    
    def get_chat_history(self):
        messages = self.chat_history.messages
        if not messages:
            return "No conversation history yet."
        
        formatted_history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_history.append(f"You: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted_history.append(f"CricketBot: {msg.content}")
        
        return "\n\n".join(formatted_history)

# Initialize the bot
cricket_bot = CricketExpertBot()

# Create Gradio interface
def create_gradio_interface():
    
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="Cricket Expert Chatbot"
    ) as ui:
        
        gr.Markdown("# 🏏 Cricket Expert Chatbot")
        gr.Markdown("### Powered by LangChain RAG Pipeline")
        gr.Markdown("Ask me anything about cricket players, statistics, records, and more!")
        
        # Chat interface
        chatbot = gr.Chatbot(
            value=[["", "Hello, I am CricketBot, your cricket expert. How can I help you today?"]],
            height=500,
            show_label=False
        )
        
        with gr.Row():
            with gr.Column(scale=4):
                msg = gr.Textbox(
                    placeholder="Ask me about cricket players...",
                    label="Your Question",
                    lines=2
                )
            with gr.Column(scale=1):
                submit_btn = gr.Button("Submit", variant="primary")
                clear_btn = gr.Button("Clear History", variant="secondary")
        
        def respond(message, chat_history):
            if message.strip() == "":
                return "", chat_history
            
            bot_response_text = cricket_bot.process_user_input(message)
            
            
            gradio_chat_history = []
            for m in cricket_bot.chat_history.messages:
                if isinstance(m, HumanMessage):
                    gradio_chat_history.append([m.content, None])
                elif isinstance(m, AIMessage):
                    if gradio_chat_history and gradio_chat_history[-1][1] is None:
                        gradio_chat_history[-1][1] = m.content
                    else: 
                         gradio_chat_history.append([None, m.content])

            if not gradio_chat_history or gradio_chat_history[-1][1] is not None:
                 gradio_chat_history.append([message, bot_response_text])

            return "", gradio_chat_history
        
        def clear_all():
            cricket_bot.clear_history()
            return "", [["", "Hello, I am CricketBot, your cricket expert. How can I help you today?"]]
        
        
        submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear_btn.click(clear_all, [], [msg, chatbot])
    
    return ui

def main():
    print("Starting Cricket Expert Chatbot...")
    
    ui = create_gradio_interface()

    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )

if __name__ == "__main__":
    main()