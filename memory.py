from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, ConversationBufferWindowMemory

def get_memory_types():
    """
    Returns a dictionary of available memory types.
    
    Returns:
        dict: A dictionary of memory type descriptions
    """
    return {
        "Buffer Memory": "Stores the full conversation history",
        "Summary Memory": "Summarizes conversation history to save tokens",
        "Window Memory": "Only remembers the most recent messages"
    }

def create_memory(memory_type, llm=None, k=5):
    """
    Creates a memory instance based on the specified type.
    
    Args:
        memory_type (str): The type of memory to create
        llm: The language model to use (required for Summary Memory)
        k (int): The number of messages to remember for Window Memory
        
    Returns:
        Memory: A LangChain memory instance
    """
    if memory_type == "Buffer Memory":
        return ConversationBufferMemory(ai_prefix="AI Assistant", human_prefix="Human")
    
    elif memory_type == "Summary Memory":
        if llm is None:
            raise ValueError("LLM must be provided for Summary Memory")
        return ConversationSummaryMemory(llm=llm, ai_prefix="AI Assistant", human_prefix="Human")
    
    elif memory_type == "Window Memory":
        return ConversationBufferWindowMemory(k=k, ai_prefix="AI Assistant", human_prefix="Human")
    
    else:
        raise ValueError(f"Unknown memory type: {memory_type}") 