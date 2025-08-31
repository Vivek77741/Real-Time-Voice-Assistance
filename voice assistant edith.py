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

GEMINI_API_KEY = ''
SERPAPI_API_KEY = ''
GROQ_API_KEY = ''

genai.configure(api_key=GEMINI_API_KEY)
recognizer = sr.Recognizer()
client = Groq(api_key=GROQ_API_KEY)

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
        if method == 'lex_rank':
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
            for res in detailed_results[:3]:
                content = res.get('snippet', '')
                if content:
                    summary = self.summarize(content, method=method)
                    summaries.append(summary)
            return ' '.join(summaries)
        return "No results found."

def get_answer_box(query):
    print("Parsed query: ", query)
    search = GoogleSearch({
        "q": query,
        "api_key": SERPAPI_API_KEY
    })
    result = search.get_dict()
    return result.get('answer_box')

def is_real_time_query(query):
    real_time_keywords = [
        'score', 'price', 'weather', 'live', 'current', 'today', 'result', 'now', 'case', 'issue', 'play',
        'update', 'news', 'stock', 'temperature', 'traffic', 'forecast', 'playing', 'will', 'movie', 'collections',
        'match', 'standings', 'alert', 'report', 'happening', 'breaking', '2024', '2025', '2023', 'ranking',
        'event', 'recent', 'trending', 'live feed', 'scorecard', 'currently',
        'tomorrow', 'yesterday', 'latest', 'instant', 'ongoing', 'real-time', 'daily', 'earthquake',
        'right now', 'immediate', 'hot', 'flash', 'buzz', 'minutes ago',
        'developing', 'direct', 'real-time update', 'headline', 'minute-by-minute', 'continuous',
        'streaming', 'live broadcast', 'now happening', 'incident', 'up-to-date', 'emergency',
        'news flash', 'quick update', 'real-time news', 'recently', 'future', 'acting',
        'up-to-the-minute', 'breaking news', 'instant update', 'real-time feed', 'moment-to-moment', 'till date', 'suggest',
        "headlines", "announcement", "just in", "game", "player", "tournament",
        "highlight", "injury", "draft", "investment", "cryptocurrency", "commodity", "exchange",
        "earnings", "storm", "rain", "hurricane", "snow", "flood", "warning", "conditions",
        "election", "candidate", "campaign", "vote", "policy", "legislation", "debate",
        "referendum", "scandal", "rally", "release", "patch", "launch", "gadget",
        "software", "hardware", "feature", "premiere", "show", "episode", "trailer",
        "casting", "award", "review", "gossip", "season", "viral", "post", "tweet",
        "hashtag", "share", "like", "comment", "follow", "engagement", "outbreak",
        "vaccine", "treatment", "study", "research", "trial", "symptoms", "diagnosis",
        "health", "wellness", "summit", "treaty", "conflict", "agreement", "diplomacy",
        "foreign", "relations", "ambassador", "sanctions", "crisis", "merger",
        "acquisition", "strategy", "revenue", "growth", "partnership", "deal",
        "initiative", "expansion", "festival", "concert", "travel", "dining",
        "fashion", "style", "trend", "recommendation", "investigation", "arrest",
        "suspect", "trial", "charge", "conviction", "witness", "evidence",
        "ceremony", "milestone", "achievement", "message", "communication",
        "reaction", "response", "follow-up", "challenge", "advocacy",
        "organization", "solution", "collaboration", "workshop", "discussion",
        "panel", "conference", "symposium", "expo", "seminar", "webinar",
        "meeting", 'president', 'minister', 'india', "debut", "phase",
        "occasion", "gathering", "reunion", "presentation", "showcase",
        "contest", "competition", "race", "league", "playoff", "charity",
        "fundraiser", "auction", "raffle", "volunteer", "sponsorship",
        "service", "project", "outreach", "survey", "poll", "feedback",
        "statistics", "analysis", "forum", "meetup", "networking",
        "roundtable", "clinic", "training", "certification", "course",
        "lecture", "class", "demonstration", "open house", "fair",
        "carnival", "market", "bazaar", "party", "breakfast", "celebration",
        "recognition", "honor", "tribute", "memorial", "remembrance",
        "observance", "commemoration", "dedication", "exhibition",
        "display", "performance", "act", "piece", "gig", "session",
        "recital", "preview", "sneak peek", "special", "edition", "version",
        "upgrade", "enhancement", "improvement", "addition", "development",
        "evolution", "transformation"
    ]
    return any(word in query.lower() for word in real_time_keywords)

def get_answer_from_groq(prompt):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192"
        )
        answer = chat_completion.choices[0].message.content
        return answer.strip()
    except Exception as e:
        return f"Sorry, there was a problem retrieving the answer: {e}"

def generate_friendly_response(prompt, data):
    try:
        response = genai.GenerativeModel('gemini-1.5-flash').generate_content(
            f"""
            Based on this information: {json.dumps(data)[:500]}
            and this question: {prompt}
            respond to the user in a friendly manner.
            """,
        )
        return response.candidates[0].content.parts[0].text.strip()
    except Exception as e:
        return f"Sorry, there was an issue generating a friendly response: {e}"

class SpeakingThread(QThread):
    finished_speaking = pyqtSignal()

    def __init__(self, text_to_speak, parent=None):
        super().__init__(parent)
        self.text_to_speak = text_to_speak
        self.engine = None
        self._is_running = True

    def run(self):
        self.engine = pyttsx3.init()
        self.engine.say(self.text_to_speak)
        self.engine.runAndWait()
        self.finished_speaking.emit()

    def stop(self):
        if self.engine:
            self.engine.stop()
        self._is_running = False

class ListeningThread(QThread):
    update_status = pyqtSignal(str, str)
    add_conversation_box = pyqtSignal(str, str)
    start_speaking = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_running = True

    def run(self):
        while self._is_running:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source)
                self.update_status.emit("Listening...", "user")
                try:
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                except sr.WaitTimeoutError:
                    if not self._is_running:
                        break
                    continue

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

                self.update_status.emit(response, "assistant")
                self.add_conversation_box.emit(response, "assistant")
                self.start_speaking.emit(response.replace('*', ''))

            except sr.UnknownValueError:
                self.update_status.emit("Sorry, I did not catch that.", "assistant")
            except sr.RequestError as e:
                self.update_status.emit(f"Could not request results; {e}", "assistant")
            except Exception as e:
                self.update_status.emit(f"An error occurred: {e}", "assistant")

    def stop(self):
        self._is_running = False

class VoiceAssistantApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.setWindowTitle("Voice Assistant")
        self.setGeometry(100, 100, 900, 700)
        self.listening_thread = None
        self.speaking_thread = None

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
        self.setGradientBackground()

    def setGradientBackground(self):
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, QColor(135, 206, 235))  # Sky Blue
        gradient.setColorAt(1.0, QColor(25, 25, 112))  # Midnight Blue
        palette = QPalette()
        palette.setBrush(QPalette.Window, QBrush(gradient))
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
            self.listening_thread.start_speaking.connect(self.speakText)
            self.listening_thread.start()
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)

    def stopListening(self):
        if self.listening_thread and self.listening_thread.isRunning():
            self.listening_thread.stop()
            self.listening_thread.wait()
        self.updateStatus("Stopped listening", "stopped")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def stopResponse(self):
        if self.speaking_thread and self.speaking_thread.isRunning():
            self.speaking_thread.stop()

    def updateStatus(self, text, role):
        self.status_label.setText(text)

    def addConversationBox(self, text, role):
        box_color = QColor("#e0f7fa") if role == "assistant" else QColor("#fff9c4")
        frame = QFrame(self.scroll_area_widget)
        frame.setStyleSheet(f"background-color: {box_color.name()}; border-radius: 10px; padding: 10px;")
        frame.setFrameShape(QFrame.StyledPanel)
        frame_layout = QVBoxLayout(frame)

        label = QLabel(text, frame)
        label.setWordWrap(True)
        frame_layout.addWidget(label)

        self.scroll_area_layout.insertWidget(self.scroll_area_layout.count() - 1, frame)
        self.scrollToBottom()

    def speakText(self, text):
        if self.speaking_thread and self.speaking_thread.isRunning():
            self.speaking_thread.stop()
            self.speaking_thread.wait()

        self.speaking_thread = SpeakingThread(text)
        self.speaking_thread.finished.connect(self.on_speaking_finished)
        self.speaking_thread.start()

    def on_speaking_finished(self):
        self.updateStatus("Ready for your next question.", "assistant")

    def scrollToBottom(self):
        self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())

    def closeEvent(self, event):
        self.stopListening()
        self.stopResponse()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    voice_assistant = VoiceAssistantApp()
    voice_assistant.show()
    sys.exit(app.exec_())
