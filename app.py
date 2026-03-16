import streamlit as st
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.agent.graph import build_agent_graph
from src.utils import get_secrets

st.set_page_config(
    page_title="Policy Document Q&A",
    page_icon="📄",
    layout="wide"
)

st.title("Policy Document Q&A System")
st.write("Ask questions about your policy documents and get AI-powered answers.")

# Initialize session state
if 'agent' not in st.session_state:
    with st.spinner("Loading AI agent..."):
        try:
            st.session_state.agent = build_agent_graph()
            st.success("AI agent loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load AI agent: {str(e)}")
            st.stop()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


# Main chat interface
st.header("Ask a Question")

# Get question from sidebar example or text input
current_question = getattr(st.session_state, 'current_question', '')
question = st.text_input(
    "Enter your question:", 
    value=current_question,
    placeholder="What policies are discussed in the documents?",
    key="question_input"
)

# Clear the current_question after using it
if hasattr(st.session_state, 'current_question'):
    del st.session_state.current_question

col1, col2 = st.columns([1, 4])
with col1:
    ask_button = st.button(
        "Ask", type="primary",
        use_container_width=True,
        disabled=st.session_state.get("loading", False) or not question.strip()
        )
with col2:
    if st.button("Clear History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

if ask_button and not st.session_state.get("loading", False):
    if question.strip():
        st.session_state.loading = True
        with st.spinner("Searching documents and generating answer..."):
            try:
                result = st.session_state.agent.invoke({"question": question})
                answer = result.get("answer", "No answer generated")
                context = result.get("context", [])
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer,
                    "context": context
                })
                
                st.session_state.question_input = ""
                st.rerun()
                
            except Exception as e:
                if "cannot be modified after the widget" not in str(e):
                    st.error(f"Error: {str(e)}")
            finally:
                st.session_state.loading = False
        

# Display chat history
if st.session_state.chat_history:
    st.header("📋 Conversation History")
    
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.container():
            st.markdown("---")
            
            # Question
            st.markdown(f"**Question {len(st.session_state.chat_history) - i}:**")
            st.info(chat['question'])
            
            # Answer
            st.markdown("**Answer:**")
            st.success(chat['answer'])
            
            # Sources
            if chat.get('context'):
                with st.expander(f"📚 View Top 3 Sources ({len(chat['context'])} found)"):
                    for j, doc in enumerate(chat['context'][:3]):  # Show top 3 sources
                        # ColBERT doesn't use just cosine similarity for score
                        # score = doc.metadata.get('score', 0)
                        # st.markdown(f"**Source {j+1}** (Relevance: {score:.1%})")
                        
                        # Show metadata if available
                        metadata = doc.metadata
                        if 'document_id' in metadata:
                            st.caption(f"Document ID: {metadata['document_id']}")
                        
                        # Show content preview
                        content_preview = doc.page_content[:400]
                        if len(doc.page_content) > 400:
                            content_preview += "..."
                        
                        st.text_area(
                            f"Content Preview {j+1}:",
                            content_preview,
                            height=100,
                            key=f"source_{i}_{j}",
                            disabled=True
                        )
                        
                        if j < len(chat['context'][:3]) - 1:
                            st.markdown("---")

else:
    st.info("Welcome! Ask your first question about the policy documents to get started.")

# Footer
st.markdown("---")
