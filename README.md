# Real-Time-Voice-Assistance
# Voice Assistant App

An interactive, AI-powered Voice Assistant desktop application built with Python and PyQt5. Seamlessly integrates voice recognition, natural language AI, web search, and real-time information retrieval to deliver an engaging assistant experience.



## ✨ Features

- **Voice Recognition:** Speak naturally—your queries are transcribed using Google's Speech Recognition.
- **Conversational Responses:** Get answers from advanced AI models (Groq Llama3, Gemini, etc.).
- **Smart Real-Time Detection:** Instantly detects when your query is about current events, news, weather, or live scores and fetches real-time data using SerpAPI.
- **Summarization:** Summarizes web search results using multiple algorithms (LSA, LexRank, Luhn, Edmundson, TextRank).
- **Text-to-Speech:** Replies are spoken aloud using `pyttsx3` for a hands-free experience.
- **Modern UI:** Beautiful gradient-themed PyQt5 interface with visually distinct conversation bubbles for user and assistant.
- **Customizable:** Easily extendable with new summarization models or AI backends.



#### Dependencies

- `PyQt5`
- `speechrecognition`
- `pyttsx3`
- `requests`
- `google-generativeai`
- `serpapi`
- `groq`
- `sumy`

### Get your API keys from these


> 💡 [Get your keys:](https://platform.openai.com/account/api-keys)  
> - [Gemini API](https://ai.google.dev/)
> - [SerpAPI](https://serpapi.com/)
> - [Groq API](https://console.groq.com/)





## 📝 Usage

- **Start Listening:** Click "Start Listening" and ask your question.
- **Speak Freely:** The assistant will transcribe, process, and respond (both visually and with voice).
- **Stop Listening:** Click "Stop Listening" to pause.
- **Stop Response:** Click "Stop Response" to interrupt text-to-speech.

---

## ⚙️ Customization

- **Add More Summarizers:** Easily add new summarization strategies via the `SearchAndSummarize` class.
- **Change AI Models:** Switch or add models in `get_answer_from_groq` or `generate_friendly_response`.
- **UI Tweaks:** Modify colors, gradients, and layout in `VoiceAssistantApp`


Pull requests, issues, and suggestions are welcome!  
Feel free to create an issue or PR on [GitHub](https://github.com/your-username/voice-assistant-app).
