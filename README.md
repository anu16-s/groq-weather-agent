# 🌤️ Breeze – AI-Powered Weather Agent

Breeze is an AI-powered weather assistant built using **Python, Streamlit, and the OpenAI Agents SDK**. It allows users to ask weather-related questions through a simple and interactive chat interface.

The application uses Large Language Models (LLMs) through compatible API providers such as Groq, Gemini, and Claude to understand user queries and provide weather information.

## ✨ Features

- **AI-Powered Chat:** Ask weather-related questions in natural language.
- **Multiple LLM Providers:** Supports Groq, Gemini, and Claude.
- **Weather Information:** Retrieve weather details using a weather API.
- **Interactive Interface:** Built with Streamlit for an easy-to-use experience.
- **Agent-Based Architecture:** Uses the OpenAI Agents SDK to manage AI agent interactions.
- **Environment Variables:** Keeps API keys and sensitive credentials outside the source code.
- **Data Visualization:** Uses Altair for weather-related visualizations.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Web application interface |
| Groq API | LLM provider |
| OpenAI Agents SDK | AI agent framework |
| OpenAI Python SDK | LLM API integration |
| Requests | API requests |
| Pandas | Data handling |
| Altair | Data visualization |
| python-dotenv | Environment variable management |

## 📁 Project Structure

```text
groq-weather-agent/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Project dependencies
├── .gitignore              # Files excluded from Git
├── .env                    # API keys and configuration (not uploaded)
│
└── .streamlit/
    └── config.toml         # Streamlit configuration
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/anu16-s/groq-weather-agent.git
```

### 2. Navigate to the Project Folder

```bash
cd groq-weather-agent
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure Environment Variables

Create a `.env` file in the project root directory.

Add your API credentials:

```env
LLM_PROVIDER=groq

GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b

GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash

ANTHROPIC_API_KEY=your_anthropic_api_key
CLAUDE_MODEL=your_claude_model

WEATHER_API_KEY=your_weather_api_key
```

Replace the placeholder values with your actual API keys.

**Important:** Never upload your `.env` file or share your API keys publicly.

### 7. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## 🤖 Supported AI Providers

Breeze supports multiple LLM providers through environment-based configuration.

| Provider | Configuration |
|---|---|
| Groq | `LLM_PROVIDER=groq` |
| Gemini | `LLM_PROVIDER=gemini` |
| Claude | `LLM_PROVIDER=claude` |

Set the required API key and model name in your `.env` file before running the application.

## 🔐 Security

- API keys are stored in environment variables.
- The `.env` file is excluded from Git using `.gitignore`.
- Never hardcode API credentials in Python files.
- Use your own API keys to run the application.

## 🚀 Future Enhancements

- Add weather forecasts for multiple days.
- Support additional cities and locations.
- Improve weather data visualization.
- Add voice-based weather queries.
- Deploy the application online.

## 👩‍💻 Author

**Anushka Shinde**

GitHub: [anu16-s](https://github.com/anu16-s)

Project Repository: [Groq Weather Agent](https://github.com/anu16-s/groq-weather-agent)

---

⭐ If you find this project interesting, feel free to explore the repository!
