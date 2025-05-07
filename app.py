import streamlit as st
from dotenv import load_dotenv
import os
from langchain.chains import ConversationChain
from utils import create_prompt_template, get_personality_templates, truncate_message, estimate_tokens
from models import get_available_models, create_llm
from memory import get_memory_types, create_memory

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(page_title="AI Chatbot with Memory", page_icon="🤖")

# Create a header
st.title("🤖 AI Chatbot with Memory")
st.subheader("Chat with an AI assistant that remembers your conversation")

# Add sidebar with information
with st.sidebar:
    st.title("About")
    st.markdown(
        "This chatbot uses OpenAI's LLM, LangChain for memory, "
        "and Streamlit for the web interface."
    )
    
    # OpenAI API Key input
    api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    
    # Add link to get OpenAI API key
    st.markdown(
        """
        <small>Don't have an API key? 
        [Get one here](https://platform.openai.com/api-keys)</small>
        """, 
        unsafe_allow_html=True
    )
    
    # Model selection
    st.subheader("Model Settings")
    available_models = get_available_models()
    model_display_name = st.selectbox(
        "Select AI Model:",
        list(available_models.keys()),
        index=0  # Default to first model (GPT-3.5 Turbo)
    )
    model_name = available_models[model_display_name]
    
    if model_display_name != "GPT-3.5 Turbo":
        st.warning(f"Note: {model_display_name} requires special access in your OpenAI account.")
    
    # Temperature slider
    temperature = st.slider(
        "Temperature:", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.7, 
        step=0.1,
        help="Higher values make output more random, lower values more deterministic"
    )
    
    # Memory selection
    st.subheader("Memory Settings")
    memory_type = st.selectbox(
        "Select Memory Type:",
        list(get_memory_types().keys()),
        help="How the chatbot remembers conversation history"
    )
    
    # Window size for window memory
    window_size = 5
    if memory_type == "Window Memory":
        window_size = st.slider(
            "Window Size:", 
            min_value=1, 
            max_value=10, 
            value=5,
            help="Number of recent messages to remember"
        )
    
    # Memory type info
    st.info(get_memory_types()[memory_type])
    
    # Personality selection
    st.subheader("Personality Settings")
    personality = st.selectbox(
        "Select chatbot personality:",
        list(get_personality_templates().keys())
    )
    
    # Token management
    st.subheader("Token Management")
    max_message_length = st.slider(
        "Max Message Length:", 
        min_value=100, 
        max_value=2000, 
        value=500,
        step=100,
        help="Longer messages use more tokens from your quota"
    )
    
    # Show estimated token usage
    if "messages" in st.session_state and st.session_state.messages:
        total_chars = sum(len(msg["content"]) for msg in st.session_state.messages)
        estimated_tokens = estimate_tokens(total_chars)
        st.write(f"Estimated token usage: ~{estimated_tokens} tokens")
        
        # Warn if token usage is high
        if estimated_tokens > 2000:
            st.warning("High token usage. Consider clearing conversation to avoid quota issues.")
    
    # Clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.memory_initialized = False
        st.experimental_rerun()

# Initialize session states for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize memory on first run or after clearing
if "memory_initialized" not in st.session_state or not st.session_state.memory_initialized:
    # We'll initialize memory when we have the API key and model
    st.session_state.memory_initialized = False

# Display the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Get user input
if prompt := st.chat_input("What's on your mind?"):
    # Only proceed if API key is provided
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar to continue.")
        st.stop()
    
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Truncate the prompt if it's too long
    truncated_prompt = truncate_message(prompt, max_message_length)
    if truncated_prompt != prompt:
        st.warning(f"Your message was truncated to {max_message_length} characters to save tokens.")
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": truncated_prompt})
    with st.chat_message("user"):
        st.write(truncated_prompt)
    
    try:
        # Create language model with selected parameters
        llm = create_llm(model_name, temperature)
        
        # Initialize memory if not already done
        if not st.session_state.memory_initialized:
            st.session_state.memory = create_memory(memory_type, llm, window_size)
            st.session_state.memory_initialized = True
        
        # Create prompt template based on personality
        prompt_template = create_prompt_template(personality)
        
        # Create conversation chain
        conversation = ConversationChain(
            llm=llm,
            prompt=prompt_template,
            memory=st.session_state.memory,
            verbose=False
        )
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner(f"Thinking using {model_display_name}..."):
                response = conversation.predict(input=truncated_prompt)
                
                # Truncate response if needed to manage token usage
                truncated_response = truncate_message(response, max_message_length)
                if truncated_response != response:
                    st.info("Response was truncated to save tokens.")
                
                st.write(truncated_response)
                # Add AI response to chat history
                st.session_state.messages.append({"role": "assistant", "content": truncated_response})
    
    except Exception as e:
        error_msg = str(e)
        with st.chat_message("assistant"):
            st.error(f"Error: {error_msg}")
            
            if "model_not_found" in error_msg.lower() or "access" in error_msg.lower():
                st.warning(f"You don't have access to {model_display_name}. Please select a different model or check your OpenAI account settings.")
                # Attempt fallback to GPT-3.5 Turbo
                st.info("Attempting to use GPT-3.5 Turbo instead...")
                try:
                    fallback_model = "gpt-3.5-turbo"
                    llm = create_llm(fallback_model, temperature)
                    
                    # Initialize memory if not already done
                    if not st.session_state.memory_initialized:
                        st.session_state.memory = create_memory(memory_type, llm, window_size)
                        st.session_state.memory_initialized = True
                    
                    # Create conversation chain
                    conversation = ConversationChain(
                        llm=llm,
                        prompt=prompt_template,
                        memory=st.session_state.memory,
                        verbose=False
                    )
                    
                    response = conversation.predict(input=truncated_prompt)
                    truncated_response = truncate_message(response, max_message_length)
                    st.write(truncated_response)
                    # Add AI response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": truncated_response})
                except Exception as fallback_error:
                    st.error(f"Fallback also failed: {str(fallback_error)}")
            
            elif "insufficient_quota" in error_msg.lower() or "exceeded your current quota" in error_msg.lower():
                st.warning("""
                Your OpenAI API key has run out of credits or exceeded its quota. You can:
                1. Check your [OpenAI billing settings](https://platform.openai.com/account/billing/overview)
                2. Add a payment method or purchase more credits
                3. Use a different API key
                """)
                st.markdown("For more information, visit [OpenAI's pricing page](https://openai.com/pricing)")
            
            elif "rate limit" in error_msg.lower():
                st.warning("""
                You're sending requests too quickly. Wait a moment and try again.
                OpenAI has rate limits to prevent abuse.
                """)
            
            else:
                st.info("If you're using GPT-4, make sure your OpenAI account has access to it.") 