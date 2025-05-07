from langchain.chat_models import ChatOpenAI

def get_available_models():
    """
    Returns a dictionary of available models with their display names and IDs.
    
    Returns:
        dict: A dictionary mapping display names to model IDs
    """
    return {
        "GPT-3.5 Turbo": "gpt-3.5-turbo",
        "GPT-4": "gpt-4",
        "GPT-4 Turbo": "gpt-4-turbo-preview"
    }

def create_llm(model_name, temperature=0.7):
    """
    Creates a language model instance based on the specified model name.
    
    Args:
        model_name (str): The name/ID of the model to use
        temperature (float, optional): The temperature for response generation. Defaults to 0.7.
        
    Returns:
        ChatOpenAI: A LangChain ChatOpenAI instance
    """
    return ChatOpenAI(temperature=temperature, model_name=model_name) 