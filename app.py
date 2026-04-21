"""
Streamlit Interface for ArXiv RAG Agent
"""

import streamlit as st
import os
from datetime import datetime

# Must be first Streamlit command
st.set_page_config(
    page_title="ArXiv RAG Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .source-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .status-badge-supported {
        background-color: #d4edda;
        color: #155724;
        padding: 0.3rem 0.6rem;
        border-radius: 4px;
        font-weight: bold;
    }
    .status-badge-partial {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.3rem 0.6rem;
        border-radius: 4px;
        font-weight: bold;
    }
    .status-badge-not {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.3rem 0.6rem;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session():
    defaults = {
        'chat_history': [],
        'agent_initialized': False,
        'rag_agent': None,
        'total_queries': 0,
        'supported_count': 0,
        'total_response_time': 0,
        'sidebar_config': {
            'top_k': 5,
            'temperature': 0.3,
            'max_tokens': 300,
            'date_filter': False
        }
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session()

# Import here to avoid loading before initialization
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rag_core import RAGAgent


def render_sidebar():
    """Sidebar configuration."""
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Model Parameters
        st.markdown("#### 🔧 Model Parameters")
        st.session_state.sidebar_config['top_k'] = st.slider(
            "Top-K Retrieved Documents", 1, 10, 
            st.session_state.sidebar_config['top_k']
        )
        st.session_state.sidebar_config['temperature'] = st.slider(
            "Temperature", 0.0, 1.0,
            st.session_state.sidebar_config['temperature'], 0.1
        )
        st.session_state.sidebar_config['max_tokens'] = st.slider(
            "Max Output Tokens", 100, 500,
            st.session_state.sidebar_config['max_tokens'], 50
        )
        
        # Filters
        st.markdown("#### 🔍 Filters")
        st.session_state.sidebar_config['date_filter'] = st.checkbox(
            "Recent papers only (last 30 days)",
            st.session_state.sidebar_config['date_filter']
        )
        
        st.markdown("---")
        
        # System Status
        st.markdown("#### 📊 System Status")
        if st.session_state.agent_initialized:
            st.success("✅ System Ready")
            st.info(f"Queries: {st.session_state.total_queries}")
            if st.session_state.total_queries > 0:
                avg_time = st.session_state.total_response_time / st.session_state.total_queries
                st.info(f"Avg Response: {avg_time:.2f}s")
        else:
            st.error("❌ System Not Initialized")
        
        # Initialize Button
        if st.button("🚀 Initialize System", type="primary"):
            with st.spinner("Loading models..."):
                try:
                    agent = RAGAgent()
                    agent.initialize()
                    st.session_state.rag_agent = agent
                    st.session_state.agent_initialized = True
                    st.success("System initialized successfully!")
                except Exception as e:
                    st.error(f"Initialization failed: {str(e)}")
        
        st.markdown("---")
        
        # Clear Chat
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.total_queries = 0
            st.session_state.supported_count = 0
            st.session_state.total_response_time = 0
            st.success("Chat cleared!")
            st.rerun()


def process_query(query: str):
    """Process user query."""
    if not st.session_state.agent_initialized:
        st.error("Please initialize the system first!")
        return
    
    agent = st.session_state.rag_agent
    config = st.session_state.sidebar_config
    
    with st.spinner("🔍 Searching..."):
        result = agent.query(
            query, 
            top_k=config['top_k']
        )
    
    # Update metrics
    st.session_state.total_queries += 1
    st.session_state.total_response_time += result['total_time']
    if result['status'] == 'Supported':
        st.session_state.supported_count += 1
    
    # Add to history
    st.session_state.chat_history.append({
        'question': query,
        'result': result,
        'timestamp': datetime.now().isoformat()
    })
    
    st.rerun()


def render_chat():
    """Main chat interface."""
    st.markdown('<div class="main-header">🤖 ArXiv RAG Agent</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask questions about AI research papers</div>', 
                unsafe_allow_html=True)
    
    # Query input
    query = st.text_input(
        "Your question:",
        placeholder="e.g., What are recent advances in transformer architectures?",
        key="query_input"
    )
    
    col1, _ = st.columns([1, 4])
    with col1:
        if st.button("🔍 Search", type="primary"):
            if query.strip():
                process_query(query)
            else:
                st.warning("Please enter a question.")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### 💬 Conversation")
        
        for chat in reversed(st.session_state.chat_history):
            # User message
            with st.chat_message("user"):
                st.write(chat['question'])
            
            # Assistant response
            with st.chat_message("assistant"):
                result = chat['result']
                
                # Answer
                st.write(result['answer'])
                
                # Status badge
                status = result['status']
                if status == 'Supported':
                    st.markdown(f'<span class="status-badge-supported">{status}</span>', 
                              unsafe_allow_html=True)
                elif status == 'Partially Supported':
                    st.markdown(f'<span class="status-badge-partial">{status}</span>', 
                              unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="status-badge-not">{status}</span>', 
                              unsafe_allow_html=True)
                
                # Metrics
                cols = st.columns(4)
                cols[0].metric("Confidence", f"{result['confidence']:.2f}")
                cols[1].metric("Response Time", f"{result['total_time']:.2f}s")
                cols[2].metric("Sources", len(result['sources']))
                cols[3].metric("Tokens", result.get('generated_tokens', 0))
                
                # Sources expander
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(result['sources']):
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>[{i+1}] {source.get('title', 'Unknown')}</strong><br>
                            <small>Doc ID: {source.get('doc_id', 'N/A')} | 
                            Similarity: {source.get('similarity', 0):.3f}</small>
                        </div>
                        """, unsafe_allow_html=True)


def render_analytics():
    """Analytics dashboard."""
    st.markdown("### 📊 Session Analytics")
    
    if not st.session_state.chat_history:
        st.info("No data yet. Start asking questions!")
        return
    
    import pandas as pd
    import plotly.express as px
    
    # Prepare data
    data = []
    for chat in st.session_state.chat_history:
        data.append({
            'question': chat['question'][:50] + '...',
            'response_time': chat['result']['total_time'],
            'confidence': chat['result']['confidence'],
            'status': chat['result']['status'],
            'sources': len(chat['result']['sources'])
        })
    
    df = pd.DataFrame(data)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Queries", len(df))
    col2.metric("Avg Response", f"{df['response_time'].mean():.2f}s")
    col3.metric("Avg Confidence", f"{df['confidence'].mean():.2f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.line(df, y='response_time', title='Response Time Trend',
                      labels={'response_time': 'Time (s)', 'index': 'Query'})
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.pie(df, names='status', title='Answer Quality Distribution')
        st.plotly_chart(fig2, use_container_width=True)
    
    # Export
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Session Data",
        csv,
        f"rag_session_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        "text/csv"
    )


def render_about():
    """About page."""
    st.markdown("""
    ### ℹ️ About ArXiv RAG Agent
    
    A complete **Retrieval-Augmented Generation** system for AI research papers.
    
    #### 🏗️ Architecture
    - **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
    - **LLM**: `mistralai/Mistral-7B-Instruct-v0.3` (4-bit quantized)
    - **Vector Store**: ChromaDB
    - **Data**: arXiv cs.AI papers
    
    #### 🚀 Features
    - Semantic search with similarity scoring
    - Confidence-based answer verification
    - Citation tracking
    - Real-time analytics
    
    #### 📋 Usage Tips
    - Be specific in questions
    - Check source papers for details
    - Adjust Top-K for broader/narrower search
    
    **Created for COMP 741 - RAG Pipeline Project**
    """)


def main():
    render_sidebar()
    
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Analytics", "ℹ️ About"])
    
    with tab1:
        render_chat()
    with tab2:
        render_analytics()
    with tab3:
        render_about()


if __name__ == "__main__":
    main()