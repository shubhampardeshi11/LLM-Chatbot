# LLM Chatbot with Memory and Personality Simulation

This project implements an AI chatbot with memory capabilities and customizable personalities using OpenAI's language models, LangChain for memory management, and Streamlit for the web interface.

## Features

- 💬 Interactive chat interface
- 🧠 Conversation memory (remembers chat history)
- 🎭 Multiple personality options
- 🔑 Secure API key input
- 🔄 Clear conversation history option

## Requirements

- Python 3.8+
- OpenAI API key

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/llm-chatbot.git
   cd llm-chatbot
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and go to `http://localhost:8501`

3. Enter your OpenAI API key in the sidebar

4. Select a personality for the chatbot

5. Start chatting!

## Personality Options

The chatbot offers four distinct personalities:

1. **Friendly and Helpful** - Upbeat, positive, and supportive assistant
2. **Professional Expert** - Formal, thorough, and knowledgeable professional
3. **Creative and Imaginative** - Outside-the-box thinker that uses metaphors and storytelling
4. **Sarcastic Humorist** - Witty assistant with light-hearted banter and playful sarcasm

## Security Note

This application does not store your OpenAI API key. The key is only used for the current session and is stored in memory. For enhanced security, you can also set up the API key as an environment variable in a `.env` file:

```
OPENAI_API_KEY=your_openai_api_key_here
```

## License

MIT 