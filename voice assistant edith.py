import sys
import json
import os
import requests
import pyttsx3
import speech_recognition as sr
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QScrollArea, QFrame
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QPalette, QLinearGradient, QBrush
import google.generativeai as genai
from serpapi import GoogleSearch
from groq import Groq
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.edmundson import EdmundsonSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer

class SearchAndSummarize:
    def __init__(self, api_key):
        self.api_key = api_key

    def search(self, query):
        params = {
            'q': query,
            'api_key': self.api_key,
            'engine': 'google'
        }
        response = requests.get('https://serpapi.com/search', params=params)
        return response.json()

    def summarize(self, text, max_sentences=2, method='lsa'):
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        if method == 'lsa':
            summarizer = LsaSummarizer()
        elif method == 'lex_rank':
            summarizer = LexRankSummarizer()
        elif method == 'luhn':
            summarizer = LuhnSummarizer()
        elif method == 'edmundson':
            summarizer = EdmundsonSummarizer()
        elif method == 'text_rank':
            summarizer = TextRankSummarizer()
        else:
            summarizer = LsaSummarizer() 
        summary = summarizer(parser.document, max_sentences)
        return ' '.join([str(sentence) for sentence in summary])

    def get_summary(self, query, method='lsa'):
        result = self.search(query)
        if 'organic_results' in result:
            detailed_results = result['organic_results']
            summaries = []
            for res in detailed_results[:3]:  # Get top 3 results
                content = res.get('snippet', '')  # Using snippet for simplicity
                if content:
                    summary = self.summarize(content, method=method)
                    summaries.append(summary)
            return ' '.join(summaries)
        return "No results found."

# Configure API keys
GEMINI_API_KEY = 'AIzaSyBdAuI0a0fe82_xiFXFcNQDg4ahIIhvRtY'
SERPAPI_API_KEY = '1c02b4f17797cf73b553a9d4c2c02a90d1ba045c561ec4ba7b21a20adbd0ac04'
GROQ_API_KEY = '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'

# Initialize APIs
genai.configure(api_key=GEMINI_API_KEY)
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Helper Functions
def get_answer_box(query):
    print("Parsed query: ", query)
    search = GoogleSearch({
        "q": query,
        "api_key": SERPAPI_API_KEY
    })
    result = search.get_dict()

    if 'answer_box' not in result:
        return None  # Return None if no answer box is found
    
    return result['answer_box']

def is_real_time_query(query):
    real_time_keywords = [
        'score', 'price', 'weather', 'live', 'current', 'today', 'result',
        'update', 'news', 'stock', 'temperature', 'traffic', 'forecast',
        'match', 'standings', 'alert', 'report', 'happening', 'breaking',
        'event', 'now', 'recent', 'trending', 'live feed', 'scorecard',
        'tomorrow', 'yesterday', 'latest', 'instant', 'ongoing', 'real-time',
        'right now', 'immediate', 'hot', 'flash', 'buzz', 'alert', 'minutes ago',
        'ongoing', 'currently', 'developing', 'direct', 'real-time update', 'headline',
        'minute-by-minute', 'continuous', 'streaming', 'live broadcast', 'now happening',
        'up-to-date', 'emergency', 'news flash', 'quick update', 'real-time news','recently',
        'up-to-the-minute', 'breaking news', 'instant update', 'real-time feed', 'moment-to-moment','now','till date'
    ]
    return any(word in query.lower() for word in real_time_keywords)

def get_answer_from_groq(prompt):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192"  # Replace with the correct model if needed
        )
        answer = chat_completion.choices[0].message.content
        return answer.strip()
    except Exception as e:
        return f"Sorry, there was a problem retrieving the answer: {e}"

def generate_friendly_response(prompt, data):
    response = genai.GenerativeModel('gemini-1.5-flash').generate_content(
        f"""
        Based on this information: {json.dumps(data)[:500]}
        and this question: {prompt}
        respond to the user in a friendly manner.
        """,
    )
    return response.candidates[0].content.parts[0].text.strip()

class VoiceAssistantApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.setWindowTitle("Voice Assistant")
        self.setGeometry(100, 100, 900, 700)

    def initUI(self):
        layout = QVBoxLayout()
        self.status_label = QLabel("Click 'Start Listening' to begin", self)
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area_widget = QWidget()
        self.scroll_area.setWidget(self.scroll_area_widget)
        self.scroll_area_layout = QVBoxLayout(self.scroll_area_widget)
        self.scroll_area_layout.addStretch()
        layout.addWidget(self.scroll_area)

        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Listening", self)
        self.start_button.clicked.connect(self.startListening)
        button_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Listening", self)
        self.stop_button.clicked.connect(self.stopListening)
        button_layout.addWidget(self.stop_button)

        self.stop_response_button = QPushButton("Stop Response", self)
        self.stop_response_button.clicked.connect(self.stopResponse)
        button_layout.addWidget(self.stop_response_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

        self.listening_thread = None
        self.setGradientBackground()

    def setGradientBackground(self):
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, QColor(255, 0, 255))
        gradient.setColorAt(1.0, QColor(0, 255, 255))
        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(gradient))
        self.setAutoFillBackground(True)
        self.setPalette(palette)

    def resizeEvent(self, event):
        self.setGradientBackground()
        super().resizeEvent(event)

    def startListening(self):
        if not self.listening_thread or not self.listening_thread.isRunning():
            self.listening_thread = ListeningThread(self)
            self.listening_thread.update_status.connect(self.updateStatus)
            self.listening_thread.add_conversation_box.connect(self.addConversationBox)
            self.listening_thread.start()

    def stopListening(self):
        if self.listening_thread and self.listening_thread.isRunning():
            self.listening_thread.terminate()
            self.updateStatus("Stopped listening", "stopped")

    def stopResponse(self):
        if self.listening_thread:
            self.listening_thread.stop_response_flag = True
            self.stopSpeaking()

    def stopSpeaking(self):
        engine.stop()

    def updateStatus(self, text, role):
        self.status_label.setText(text)

    def addConversationBox(self, text, role):
        box_color = QColor("#c0f0c0") if role == "assistant" else QColor("#f0c0c0")
        frame = QFrame(self.scroll_area_widget)
        frame.setStyleSheet(f"background-color: {box_color.name()}; border: 1px solid #000000;")
        frame.setMinimumHeight(120)
        frame.setFrameShape(QFrame.StyledPanel)
        frame_layout = QVBoxLayout(frame)

        label = QLabel(text, frame)
        label.setWordWrap(True)
        frame_layout.addWidget(label)

        self.scroll_area_layout.insertWidget(self.scroll_area_layout.count() - 1, frame)
        self.scrollToBottom()

    def scrollToBottom(self):
        self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())

class ListeningThread(QThread):
    update_status = pyqtSignal(str, str)
    add_conversation_box = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stop_response_flag = False

    def run(self):
        while True:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source)
                self.update_status.emit("Listening...", "user")
                audio = recognizer.listen(source)

            try:
                query = recognizer.recognize_google(audio)
                self.update_status.emit(f"You said: {query}", "user")
                self.add_conversation_box.emit(query, "user")

                if is_real_time_query(query):
                    answer_box = get_answer_box(query)
                    if answer_box:
                        response = generate_friendly_response(query, answer_box)
                    else:
                        search_summary = SearchAndSummarize(SERPAPI_API_KEY).get_summary(query)
                        response = generate_friendly_response(query, search_summary)
                else:
                    response = get_answer_from_groq(query)

                if self.stop_response_flag:
                    break

                self.update_status.emit(response, "assistant")
                self.add_conversation_box.emit(response, "assistant")
                engine.say(response)
                engine.runAndWait()

                if self.stop_response_flag:
                    break
            except sr.UnknownValueError:
                self.update_status.emit("Sorry, I did not catch that.", "assistant")
            except sr.RequestError as e:
                self.update_status.emit(f"Could not request results; {e}", "assistant")
            except Exception as e:
                self.update_status.emit(f"An error occurred: {e}", "assistant")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    voice_assistant = VoiceAssistantApp()
    voice_assistant.show()
    sys.exit(app.exec_())
