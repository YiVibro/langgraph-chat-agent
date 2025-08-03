# LangGraph Conversational Agent with Tools

A structured, multi-tool conversational agent built using LangGraph and Google Gemini. Supports real-time web search (Tavily) and human input fallback, with intelligent message routing and checkpointing.

## 🔧 Setup

1. **Clone the repo**  
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```
2. **Install dependencies**
```bash
pip install -r requirements.txt
```
3. **Add API keys to .env**
```bash
GOOGLE_API_KEY=your_google_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
```
How It Works
LangGraph: Orchestrates the conversation using a stateful graph with message-based state.

Gemini via LangChain: Generates responses and decides whether to invoke tools.

Tools:

🧠 human_assistance: Fallback when unsure.

🔍 TavilySearch: Real-time search.

Routing: Automatically detects when tools are needed based on Gemini’s tool_calls.

**Example Usage**
```python
# You can run this via terminal or inside a script
from my_agent import graph  # or main.py

# Start a chat session
thread = graph.compile()
state = {"messages": [{"role": "user", "content": "Who won the latest Formula 1 race?"}]}
for s in thread.stream(state):
    print(s.get("messages")[-1].content)
```
 **Credits**
1. LangGraph

2. Google Gemini

3. Tavily
 
