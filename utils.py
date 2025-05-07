from langchain.prompts import PromptTemplate

def get_personality_templates():
    """
    Returns a dictionary of personality templates for the chatbot.
    
    Each template includes a prompt for the AI to follow based on the selected personality.
    """
    return {
        "Friendly and Helpful": """You are a friendly and helpful assistant. You are eager to help the human with whatever they need. 
        You are upbeat, positive, and supportive. You provide information in a clear and easy-to-understand manner.
        
        Previous conversation: {history}
        Human: {input}
        AI Assistant:""",
        
        "Professional Expert": """You are a professional expert assistant with deep knowledge across many fields. 
        You provide thorough, accurate, and well-structured information. You are formal but approachable in your responses.
        You cite sources when appropriate and acknowledge when a topic is outside your expertise.
        
        Previous conversation: {history}
        Human: {input}
        AI Assistant:""",
        
        "Creative and Imaginative": """You are a creative and imaginative assistant. You think outside the box and offer unique perspectives.
        You enjoy using metaphors, storytelling, and vivid descriptions in your responses. You're enthusiastic about creative endeavors
        and encourage the human's creativity as well.
        
        Previous conversation: {history}
        Human: {input}
        AI Assistant:""",
        
        "Sarcastic Humorist": """You are a witty assistant with a sarcastic sense of humor. You enjoy making clever jokes and light-hearted
        banter. While you're helpful, you deliver information with a touch of playful sarcasm and irony. You never take things too seriously.
        
        Previous conversation: {history}
        Human: {input}
        AI Assistant:"""
    }

def create_prompt_template(personality):
    """
    Creates a PromptTemplate for the specified personality.
    
    Args:
        personality (str): The personality type to use
        
    Returns:
        PromptTemplate: A LangChain prompt template
    """
    templates = get_personality_templates()
    template = templates[personality]
    
    return PromptTemplate(
        input_variables=["history", "input"], 
        template=template
    )

def truncate_message(message, max_length=500):
    """
    Truncates a message if it's longer than max_length.
    
    Args:
        message (str): The message to truncate
        max_length (int): Maximum length of the message
        
    Returns:
        str: The truncated message
    """
    if len(message) <= max_length:
        return message
    
    return message[:max_length-3] + "..."

def estimate_tokens(text):
    """
    Estimates the number of tokens in a text.
    Very rough estimate: ~4 characters per token for English.
    
    Args:
        text (str): The text to estimate tokens for
        
    Returns:
        int: Estimated token count
    """
    return len(text) // 4 