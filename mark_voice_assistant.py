import sys
import json
import os
import re
import time
import io
import queue
import subprocess
import datetime
import zoneinfo
import webbrowser
from typing import Optional, Dict, Any, List

import pyttsx3
import pythoncom
import speech_recognition as sr
import pyautogui
import win32gui
from PIL import Image

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QScrollArea, QFrame, QTextEdit, QGraphicsDropShadowEffect, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPointF, QBuffer, QIODevice
from PyQt5.QtGui import (
    QColor, QPalette, QLinearGradient, QBrush, QFont, QPainter,
    QPen, QRadialGradient, QPixmap
)

from google import genai
from google.genai import types
from serpapi import GoogleSearch
from groq import Groq

# Safety settings for GUI automation
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05

# Load environment variables from local .env file if available
_env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(_env_file):
    with open(_env_file, 'r', encoding='utf-8') as _f:
        for _line in _f:
            _line = _line.strip()
            if '=' in _line and not _line.startswith('#'):
                _k, _v = _line.split('=', 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip('"\''))

# API Keys (Loaded securely from Environment Variables or .env)
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
SERPAPI_API_KEY = os.environ.get('SERPAPI_API_KEY', '')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')

# Initialize Clients
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Initialize Speech Recognizer with anti-cutoff tuning
recognizer = sr.Recognizer()
recognizer.dynamic_energy_threshold = False
recognizer.energy_threshold = 280
recognizer.pause_threshold = 0.8          # Snappy endpointing (0.8s silence triggers processing)
recognizer.non_speaking_duration = 0.8    # Keep trailing buffers so words like 'in India' are never lost
recognizer.phrase_threshold = 0.25

# ==============================================================================
# 1. ADVANCED AUTONOMOUS TOOL ARSENAL (FULL COMPUTER-USE & VISION)
# ==============================================================================

class MarkToolArsenal:
    """Comprehensive tool suite giving MARK full laptop, web, vision, and system access."""

    TIMEZONE_MAP = {
        'india': 'Asia/Kolkata', 'ist': 'Asia/Kolkata', 'delhi': 'Asia/Kolkata', 'mumbai': 'Asia/Kolkata',
        'usa': 'America/New_York', 'new york': 'America/New_York', 'ny': 'America/New_York',
        'california': 'America/Los_Angeles', 'la': 'America/Los_Angeles', 'san francisco': 'America/Los_Angeles',
        'london': 'Europe/London', 'uk': 'Europe/London', 'gmt': 'GMT', 'utc': 'UTC',
        'dubai': 'Asia/Dubai', 'uae': 'Asia/Dubai',
        'tokyo': 'Asia/Tokyo', 'japan': 'Asia/Tokyo',
        'singapore': 'Asia/Singapore',
        'sydney': 'Australia/Sydney', 'australia': 'Australia/Sydney',
        'paris': 'Europe/Paris', 'germany': 'Europe/Berlin', 'berlin': 'Europe/Berlin'
    }

    @classmethod
    def get_world_time(cls, location: str = 'india') -> str:
        """Instant 0ms timezone lookup."""
        clean_loc = location.lower().strip()
        tz_str = 'Asia/Kolkata'
        for key, val in cls.TIMEZONE_MAP.items():
            if key in clean_loc:
                tz_str = val
                break
        try:
            tz = zoneinfo.ZoneInfo(tz_str)
            now = datetime.datetime.now(tz)
            formatted = now.strftime('%I:%M %p, %A, %B %d, %Y')
            return f"The current time in {location.title()} ({tz_str}) is {formatted}."
        except Exception as e:
            now = datetime.datetime.now()
            return f"Local time is {now.strftime('%I:%M %p, %A, %B %d, %Y')} (Error: {e})."

    @classmethod
    def capture_screen_bytes(cls) -> Optional[bytes]:
        """Captures full screen in <15ms using PyQt QBuffer with pyautogui fallback."""
        try:
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            screen = QApplication.primaryScreen()
            if screen:
                pixmap = screen.grabWindow(0)
                qbuf = QBuffer()
                qbuf.open(QIODevice.WriteOnly)
                pixmap.save(qbuf, "JPEG", 85)
                return bytes(qbuf.data())
        except Exception as e:
            print(f"[Qt Screen Grab Warning] {e}")

        # Fallback via pyautogui / PIL
        try:
            img = pyautogui.screenshot()
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return buf.getvalue()
        except Exception as e:
            print(f"[Fallback Screen Grab Error] {e}")
        return None

    @classmethod
    def screen_vision(cls, prompt: str = "Describe what is currently on the screen") -> str:
        """Multimodal screen inspection using Gemini 2.5 Flash."""
        try:
            if not gemini_client or not GEMINI_API_KEY:
                return "Gemini API key is not configured. Please set GEMINI_API_KEY in your environment or .env file."
            img_bytes = cls.capture_screen_bytes()
            if not img_bytes:
                return "Failed to capture screen buffer."

            response = gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'),
                    f"You are the visual cortex for the MARK voice assistant. The user asks: '{prompt}'. Inspect the screen and give a sharp, direct, concise answer in 1-2 sentences."
                ]
            )
            return response.text.strip()
        except Exception as e:
            return f"Screen vision error: {e}"

    @classmethod
    def click_element_on_screen(cls, description: str) -> str:
        """
        Uses Gemini 2.5 Flash visual grounding to find the exact (x, y) coordinates
        of any element on the screen (button, link, video thumbnail, text box) and clicks it.
        """
        try:
            if not gemini_client or not GEMINI_API_KEY:
                return "Gemini API key is not configured. Please set GEMINI_API_KEY in your environment or .env file."
            img_bytes = cls.capture_screen_bytes()
            if not img_bytes:
                return "Failed to grab screen for element detection."

            prompt = (
                f"Find the UI element '{description}' on this screen. "
                f"Return ONLY valid JSON with normalized coordinates between 0.0 and 1.0: "
                f'{{"x_pct": float, "y_pct": float}}'
            )

            res = gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'),
                    prompt
                ],
                config={'response_mime_type': 'application/json'}
            )

            data = json.loads(res.text.strip())
            x_pct = float(data.get("x_pct", 0.5))
            y_pct = float(data.get("y_pct", 0.5))

            screen_w, screen_h = pyautogui.size()
            target_x = int(x_pct * screen_w)
            target_y = int(y_pct * screen_h)

            pyautogui.moveTo(target_x, target_y, duration=0.3)
            pyautogui.click()
            return f"Located '{description}' at ({target_x}, {target_y}) and clicked."
        except Exception as e:
            return f"Could not locate '{description}' on screen: {e}"

    @classmethod
    def launch_app_or_url(cls, target: str) -> str:
        """
        Robust launcher for any Windows application, URL, folder, or file.
        Uses os.startfile to ensure reliable Windows app resolution.
        """
        try:
            clean = target.lower().strip()

            # Web URLs or domains
            if clean.startswith(('http://', 'https://', 'www.')) or ('.' in clean and ' ' not in clean and not clean.endswith('.exe')):
                url = target if target.startswith(('http://', 'https://')) else f"https://{target}"
                webbrowser.open(url)
                return f"Opened web URL: {url}"

            # YouTube shortcut
            if clean == 'youtube':
                webbrowser.open("https://youtube.com")
                return "Opened YouTube in default browser."

            # Known Windows Application Executables
            app_map = {
                'notepad': 'notepad.exe',
                'calculator': 'calc.exe',
                'calc': 'calc.exe',
                'explorer': 'explorer.exe',
                'files': 'explorer.exe',
                'cmd': 'cmd.exe',
                'terminal': 'wt.exe',
                'vscode': 'code',
                'vs code': 'code',
                'task manager': 'taskmgr.exe'
            }

            if clean in app_map:
                try:
                    os.startfile(app_map[clean])
                    return f"Launched {clean}."
                except Exception:
                    subprocess.Popen(app_map[clean], shell=True)
                    return f"Launched {clean}."

            # For Chrome, Edge, and other installed browsers
            if 'chrome' in clean:
                chrome_paths = [
                    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe")
                ]
                for p in chrome_paths:
                    if os.path.exists(p):
                        os.startfile(p)
                        return "Launched Google Chrome."
                os.system('start chrome')
                return "Launched Google Chrome via shell."

            if 'edge' in clean:
                os.system('start msedge')
                return "Launched Microsoft Edge."

            # Generic Windows startfile (handles any installed app, file, or directory)
            try:
                os.startfile(target)
                return f"Opened {target}."
            except Exception:
                os.system(f'start "" "{target}"')
                return f"Launched {target} via Windows Shell."

        except Exception as e:
            return f"Failed to launch '{target}': {e}"

    @classmethod
    def browser_navigation(cls, action: str) -> str:
        """
        Human-like browser control: back, forward, new_tab, close_tab, refresh,
        scroll_down, scroll_up, play_pause video, fullscreen, etc.
        """
        try:
            act = action.lower().strip()
            if act in ['back', 'go_back', 'previous']:
                pyautogui.hotkey('alt', 'left')
                return "Navigated back."
            elif act in ['forward', 'go_forward']:
                pyautogui.hotkey('alt', 'right')
                return "Navigated forward."
            elif act in ['refresh', 'reload']:
                pyautogui.press('f5')
                return "Reloaded page."
            elif act in ['new_tab', 'open_tab']:
                pyautogui.hotkey('ctrl', 't')
                return "Opened new browser tab."
            elif act in ['close_tab']:
                pyautogui.hotkey('ctrl', 'w')
                return "Closed current tab."
            elif act in ['switch_tab', 'next_tab']:
                pyautogui.hotkey('ctrl', 'tab')
                return "Switched to next tab."
            elif act in ['scroll_down', 'scroll']:
                pyautogui.scroll(-700)
                return "Scrolled down."
            elif act in ['scroll_up']:
                pyautogui.scroll(700)
                return "Scrolled up."
            elif act in ['play', 'pause', 'play_pause', 'toggle_video']:
                pyautogui.press('space')
                return "Toggled play/pause."
            elif act in ['fullscreen']:
                pyautogui.press('f')
                return "Toggled fullscreen."
            elif act in ['mute']:
                pyautogui.press('m')
                return "Toggled video mute."
            return f"Unknown browser action: {action}"
        except Exception as e:
            return f"Browser action failed: {e}"

    @classmethod
    def youtube_search_and_play(cls, query: str) -> str:
        """Searches YouTube and optionally plays the top result."""
        try:
            clean_q = query.replace(' ', '+')
            url = f"https://www.youtube.com/results?search_query={clean_q}"
            webbrowser.open(url)
            time.sleep(1.2)
            pyautogui.press('tab')
            pyautogui.press('enter')
            return f"Searched YouTube for '{query}' and initiated playback."
        except Exception as e:
            return f"YouTube action error: {e}"

    @classmethod
    def mouse_action(cls, action: str, x: int = 0, y: int = 0, button: str = 'left', clicks: int = 1) -> str:
        """Autonomously clicks, moves, or scrolls the mouse."""
        try:
            screen_w, screen_h = pyautogui.size()
            x = max(0, min(screen_w - 1, int(x)))
            y = max(0, min(screen_h - 1, int(y)))

            if action in ['click', 'press']:
                pyautogui.click(x=x, y=y, button=button, clicks=clicks)
                return f"Mouse clicked at ({x}, {y}) with {button} button."
            elif action == 'double_click':
                pyautogui.doubleClick(x=x, y=y)
                return f"Mouse double-clicked at ({x}, {y})."
            elif action == 'right_click':
                pyautogui.rightClick(x=x, y=y)
                return f"Mouse right-clicked at ({x}, {y})."
            elif action == 'move':
                pyautogui.moveTo(x=x, y=y, duration=0.2)
                return f"Mouse moved to ({x}, {y})."
            elif action == 'scroll':
                pyautogui.scroll(clicks * 100)
                return f"Scrolled mouse by {clicks * 100} units."
            return f"Unknown mouse action: {action}"
        except Exception as e:
            return f"Mouse action failed: {e}"

    @classmethod
    def keyboard_action(cls, action: str, text: str = '', keys: str = '') -> str:
        """Types text or presses key shortcuts."""
        try:
            if action == 'type':
                pyautogui.write(text, interval=0.01)
                return f"Typed: '{text}'"
            elif action in ['hotkey', 'press']:
                key_list = [k.strip().lower() for k in (keys or text).split('+')]
                pyautogui.hotkey(*key_list)
                return f"Pressed keys: {key_list}"
            return f"Unknown keyboard action: {action}"
        except Exception as e:
            return f"Keyboard action failed: {e}"

    @classmethod
    def search_web(cls, query: str) -> str:
        """Fast SerpApi search with instant answer box extraction."""
        try:
            if not SERPAPI_API_KEY:
                return "SerpApi key is not configured. Please set SERPAPI_API_KEY in your environment or .env file."
            search = GoogleSearch({
                'q': query,
                'api_key': SERPAPI_API_KEY,
                'num': 3
            })
            result = search.get_dict()

            if 'answer_box' in result:
                ab = result['answer_box']
                if 'result' in ab: return f"Instant Answer: {ab.get('result')}"
                if 'answer' in ab: return f"Instant Answer: {ab.get('answer')}"
                if 'snippet' in ab: return f"Instant Answer: {ab.get('snippet')}"

            if 'organic_results' in result and len(result['organic_results']) > 0:
                snippets = []
                for item in result['organic_results'][:2]:
                    snip = item.get('snippet', '')
                    if snip: snippets.append(f"{item.get('title', '')}: {snip}")
                if snippets: return " | ".join(snippets)

            return "No direct web search results found."
        except Exception as e:
            return f"Web search error: {e}"

    @classmethod
    def calculate_math(cls, expression: str) -> str:
        """Safely evaluates math expressions."""
        try:
            allowed = re.sub(r'[^0-9\+\-\*\/\(\)\.\% ]', '', expression)
            res = eval(allowed, {"__builtins__": None}, {})
            return f"The result of {expression} is {res}."
        except Exception as e:
            return f"Calculation error: {e}"

    @classmethod
    def get_system_info(cls) -> str:
        """Returns active foreground window and screen resolution."""
        try:
            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd)
            w, h = pyautogui.size()
            return f"Active Window: '{title}' | Screen Resolution: {w}x{h}"
        except Exception as e:
            return f"System info: {e}"


# ==============================================================================
# 2. STREAMING SENTENCE-BY-SENTENCE AUDIO WORKER (SUB-SECOND TTFA)
# ==============================================================================

class StreamingTTSWorker(QThread):
    """
    Dedicated audio worker thread with COM thread safety.
    Guarantees voice never hangs or stops after visual/inspection actions.
    """
    speaking_started = pyqtSignal(str)
    speaking_finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.audio_queue = queue.Queue()
        self.is_running = True

    def run(self):
        pythoncom.CoInitialize()
        engine = None
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 190)
            engine.setProperty('volume', 1.0)
        except Exception as e:
            print(f"[TTS Init Warning] {e}")

        while self.is_running:
            try:
                text = self.audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if not self.is_running:
                break

            if text == "__STOP__":
                if engine:
                    try: engine.stop()
                    except Exception: pass
                continue

            clean_text = text.replace('*', '').replace('#', '').replace('`', '').strip()
            if clean_text:
                self.speaking_started.emit(clean_text)
                try:
                    if engine is None:
                        engine = pyttsx3.init()
                        engine.setProperty('rate', 190)
                    engine.say(clean_text)
                    engine.runAndWait()
                except Exception as e:
                    print(f"[TTS Speak Error] {e}")
                    try:
                        engine = pyttsx3.init()
                        engine.setProperty('rate', 190)
                    except Exception:
                        pass
                self.speaking_finished.emit()

    def queue_sentence(self, sentence: str):
        if sentence.strip():
            self.audio_queue.put(sentence.strip())

    def stop_speaking(self):
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
        self.audio_queue.put("__STOP__")

    def shutdown(self):
        self.is_running = False
        self.stop_speaking()
        self.wait(800)


# ==============================================================================
# 3. MARK AGENT ORCHESTRATOR (REASONING & ROUTING)
# ==============================================================================

class MarkAgentOrchestrator:
    """
    Autonomous ReAct agent powered by Groq Qwen 27B / Gemini.
    Selects and executes tools, streams answers, and controls the laptop like a human.
    """

    SYSTEM_PROMPT = """You are MARK (Multimodal Autonomous Robotic Kernel), an elite, ultra-fast autonomous voice and computer-use AI assistant.
You have direct, full control over this laptop, screen vision, browser, and OS tools.

Available Tools:
1. `get_world_time(location)` -> Instant time for India, London, New York, Tokyo, Dubai, etc. (USE THIS FOR ANY TIME/DATE QUERY!)
2. `screen_vision(prompt)` -> Takes a screenshot right now and inspects what the user has open on screen.
3. `click_element_on_screen(description)` -> Uses vision AI to locate any button/thumbnail/link on screen and clicks it! (e.g. 'first video', 'search button', 'skip ad').
4. `launch_app(target)` -> Opens any app (chrome, youtube, notepad, vscode, calc, terminal) or URL.
5. `browser_navigation(action)` -> Controls browser: 'back', 'forward', 'new_tab', 'close_tab', 'scroll_down', 'scroll_up', 'play_pause', 'fullscreen'.
6. `youtube_search_and_play(query)` -> Searches YouTube for a video and starts playing it.
7. `mouse_action(action, x, y, button, clicks)` -> Controls mouse ('click', 'move', 'scroll', 'double_click').
8. `keyboard_action(action, text, keys)` -> Types text or presses key combos ('ctrl+c', 'enter', 'win+d').
9. `search_web(query)` -> Live web search, scores, latest news, stock prices.
10. `calculate_math(expression)` -> Evaluates mathematical / financial equations.
11. `get_system_info()` -> Gets active window and screen stats.

INSTRUCTIONS:
- If a tool is needed, your response MUST start with:
  [TOOL: tool_name(param="value")]
- For screen vision ('what is on my screen', 'look at my screen', 'read this error'):
  Use: [TOOL: screen_vision(prompt="inspect screen content")]
- For human-like clicking ('click the first video', 'click search', 'click play'):
  Use: [TOOL: click_element_on_screen(description="first video thumbnail")]
- For browser navigation ('go back', 'scroll down', 'pause the video', 'fullscreen'):
  Use: [TOOL: browser_navigation(action="back")] or [TOOL: browser_navigation(action="play_pause")]
- For YouTube video requests ('play coldplay on youtube', 'search python tutorial on youtube'):
  Use: [TOOL: youtube_search_and_play(query="coldplay viva la vida")]
- For time queries ('what is the time in India'):
  Use: [TOOL: get_world_time(location="India")]
- For launching apps ('open chrome', 'open notepad'):
  Use: [TOOL: launch_app(target="chrome")]
- If no tool is needed (conversation, explanations, coding), reply directly and punchily!
"""

    @classmethod
    def parse_and_execute_tool(cls, text: str) -> Optional[tuple[str, str, str]]:
        match = re.search(r'\[TOOL:\s*(\w+)\((.*?)\)\]', text)
        if not match:
            return None

        tool_name = match.group(1).strip()
        raw_args = match.group(2).strip()

        kwargs = {}
        for part in re.finditer(r'(\w+)\s*=\s*["\'](.*?)["\']', raw_args):
            kwargs[part.group(1)] = part.group(2)

        if not kwargs and raw_args:
            single = raw_args.strip('"\'')
            if tool_name == 'get_world_time': kwargs = {'location': single}
            elif tool_name == 'screen_vision': kwargs = {'prompt': single}
            elif tool_name == 'click_element_on_screen': kwargs = {'description': single}
            elif tool_name == 'launch_app': kwargs = {'target': single}
            elif tool_name == 'browser_navigation': kwargs = {'action': single}
            elif tool_name == 'youtube_search_and_play': kwargs = {'query': single}
            elif tool_name == 'search_web': kwargs = {'query': single}
            elif tool_name == 'calculate_math': kwargs = {'expression': single}

        result = "Tool completed."
        try:
            if tool_name == 'get_world_time':
                result = MarkToolArsenal.get_world_time(**kwargs)
            elif tool_name == 'screen_vision':
                result = MarkToolArsenal.screen_vision(**kwargs)
            elif tool_name == 'click_element_on_screen':
                result = MarkToolArsenal.click_element_on_screen(**kwargs)
            elif tool_name == 'launch_app':
                result = MarkToolArsenal.launch_app_or_url(**kwargs)
            elif tool_name == 'browser_navigation':
                result = MarkToolArsenal.browser_navigation(**kwargs)
            elif tool_name == 'youtube_search_and_play':
                result = MarkToolArsenal.youtube_search_and_play(**kwargs)
            elif tool_name == 'mouse_action':
                result = MarkToolArsenal.mouse_action(**kwargs)
            elif tool_name == 'keyboard_action':
                result = MarkToolArsenal.keyboard_action(**kwargs)
            elif tool_name == 'search_web':
                result = MarkToolArsenal.search_web(**kwargs)
            elif tool_name == 'calculate_math':
                result = MarkToolArsenal.calculate_math(**kwargs)
            elif tool_name == 'get_system_info':
                result = MarkToolArsenal.get_system_info()
            else:
                result = f"Unknown tool: {tool_name}"
        except Exception as e:
            result = f"Error executing {tool_name}: {e}"

        return (tool_name, raw_args, result)


# ==============================================================================
# 4. FUTURISTIC MARK CYBER HUD UI (ASSEMBLYAI GTM SHOWCASE QUALITY)
# ==============================================================================

class AICoreVisualizer(QWidget):
    """Animated reactive Iron Man HUD Reactor Core with multi-ring cyber pulses."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(140, 140)
        self.setMaximumSize(140, 140)
        self.state = "IDLE"
        self.angle = 0
        self.pulse = 0.0
        self.pulse_dir = 1

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(25)

    def set_state(self, state: str):
        self.state = state
        self.update()

    def update_animation(self):
        self.angle = (self.angle + 3) % 360
        self.pulse += 0.04 * self.pulse_dir
        if self.pulse > 1.0:
            self.pulse = 1.0
            self.pulse_dir = -1
        elif self.pulse < 0.0:
            self.pulse = 0.0
            self.pulse_dir = 1
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        cx = self.width() / 2
        cy = self.height() / 2

        if self.state == "LISTENING":
            core_color = QColor(0, 255, 136)
            outer_glow = QColor(0, 255, 136, 60)
        elif self.state == "THINKING":
            core_color = QColor(255, 187, 0)
            outer_glow = QColor(255, 187, 0, 80)
        elif self.state == "SPEAKING":
            core_color = QColor(168, 85, 247)
            outer_glow = QColor(168, 85, 247, 80)
        else:
            core_color = QColor(0, 240, 255)
            outer_glow = QColor(0, 240, 255, 40)

        # Outer Aura
        radius_glow = 58 + (self.pulse * 8)
        radial_grad = QRadialGradient(cx, cy, radius_glow)
        radial_grad.setColorAt(0.0, outer_glow)
        radial_grad.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.setBrush(QBrush(radial_grad))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(cx, cy), radius_glow, radius_glow)

        # Rotating Dashed Tech Ring
        pen = QPen(core_color, 2)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.drawEllipse(QPointF(cx, cy), 50, 50)

        # Rotating HUD Arcs
        pen_arc = QPen(core_color, 3)
        painter.setPen(pen_arc)
        painter.drawArc(int(cx - 42), int(cy - 42), 84, 84, int(self.angle * 16), int(70 * 16))
        painter.drawArc(int(cx - 42), int(cy - 42), 84, 84, int((self.angle + 180) * 16), int(70 * 16))

        # Core
        core_radius = 22 + (self.pulse * 5)
        core_grad = QRadialGradient(cx, cy, core_radius)
        core_grad.setColorAt(0.0, QColor(255, 255, 255))
        core_grad.setColorAt(0.6, core_color)
        core_grad.setColorAt(1.0, QColor(core_color.red(), core_color.green(), core_color.blue(), 100))
        painter.setBrush(QBrush(core_grad))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(cx, cy), core_radius, core_radius)


class MarkCyberHUD(QWidget):
    """Main MARK Cyber HUD Application Window."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MARK // Autonomous Voice & Computer-Use Agent")
        self.setGeometry(120, 80, 960, 780)
        self.setMinimumSize(850, 680)

        # Audio Worker
        self.tts_worker = StreamingTTSWorker()
        self.tts_worker.speaking_started.connect(self.on_speaking_started)
        self.tts_worker.speaking_finished.connect(self.on_speaking_finished)
        self.tts_worker.start()

        self.listener_thread = None

        self.init_ui()
        self.apply_cyber_styling()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(14)

        # 1. Header Bar
        header_frame = QFrame(self)
        header_frame.setObjectName("headerFrame")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(15, 10, 15, 10)

        self.ai_core = AICoreVisualizer(self)
        header_layout.addWidget(self.ai_core)

        info_layout = QVBoxLayout()
        title_label = QLabel("M.A.R.K.", self)
        title_label.setObjectName("hudTitle")
        info_layout.addWidget(title_label)

        sub_label = QLabel("AUTONOMOUS VOICE & LAPTOP COMPUTER-USE AGENT // ASSEMBLY AI GTM", self)
        sub_label.setObjectName("hudSubtitle")
        info_layout.addWidget(sub_label)

        self.status_badge = QLabel("STANDBY // READY FOR VOICE COMMAND", self)
        self.status_badge.setObjectName("statusBadge")
        info_layout.addWidget(self.status_badge)

        header_layout.addLayout(info_layout)
        header_layout.addStretch()

        telemetry_layout = QVBoxLayout()
        self.latency_label = QLabel("⚡ TTFA: -- ms", self)
        self.latency_label.setObjectName("telemetryItem")
        self.speed_label = QLabel("🚀 SPEED: 520 tok/s", self)
        self.speed_label.setObjectName("telemetryItem")
        self.tools_label = QLabel("🛠️ AGENT TOOLS: 11 ACTIVE", self)
        self.tools_label.setObjectName("telemetryItem")

        telemetry_layout.addWidget(self.latency_label)
        telemetry_layout.addWidget(self.speed_label)
        telemetry_layout.addWidget(self.tools_label)
        header_layout.addLayout(telemetry_layout)

        main_layout.addWidget(header_frame)

        # 2. Action Banner
        self.action_banner = QLabel("Agent Idle // Press 'Start Listening' or say a command", self)
        self.action_banner.setObjectName("actionBanner")
        self.action_banner.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.action_banner)

        # 3. Conversation Scroll Area
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setObjectName("scrollArea")

        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setContentsMargins(15, 15, 15, 15)
        self.scroll_layout.setSpacing(12)
        self.scroll_layout.addStretch()

        self.scroll_area.setWidget(self.scroll_widget)
        main_layout.addWidget(self.scroll_area)

        # 4. Action Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        self.listen_btn = QPushButton("🎙️ START LISTENING", self)
        self.listen_btn.setObjectName("listenBtn")
        self.listen_btn.clicked.connect(self.toggle_listening)
        btn_layout.addWidget(self.listen_btn)

        self.screen_btn = QPushButton("👁️ INSPECT SCREEN NOW", self)
        self.screen_btn.setObjectName("screenBtn")
        self.screen_btn.clicked.connect(self.inspect_screen_manual)
        btn_layout.addWidget(self.screen_btn)

        self.stop_btn = QPushButton("🛑 INTERRUPT / MUTE", self)
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.clicked.connect(self.stop_all_actions)
        btn_layout.addWidget(self.stop_btn)

        self.clear_btn = QPushButton("⚡ CLEAR HUD", self)
        self.clear_btn.setObjectName("clearBtn")
        self.clear_btn.clicked.connect(self.clear_chat)
        btn_layout.addWidget(self.clear_btn)

        main_layout.addLayout(btn_layout)

    def apply_cyber_styling(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #08090d;
                color: #e2e8f0;
                font-family: 'Segoe UI', -apple-system, Roboto, sans-serif;
            }
            #headerFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0f121d, stop:1 #131726);
                border: 1px solid #1e2640;
                border-radius: 12px;
            }
            #hudTitle {
                color: #00f0ff;
                font-size: 24px;
                font-weight: 800;
                letter-spacing: 3px;
            }
            #hudSubtitle {
                color: #94a3b8;
                font-size: 10px;
                font-weight: 600;
                letter-spacing: 1.5px;
            }
            #statusBadge {
                color: #00ff88;
                font-size: 12px;
                font-weight: 700;
                padding-top: 4px;
            }
            #telemetryItem {
                color: #38bdf8;
                font-size: 11px;
                font-weight: 600;
                font-family: 'Consolas', monospace;
            }
            #actionBanner {
                background-color: #111422;
                color: #38bdf8;
                border: 1px solid #1e293b;
                border-radius: 8px;
                font-size: 12px;
                font-weight: 600;
                padding: 6px;
                font-family: 'Consolas', monospace;
            }
            #scrollArea {
                background-color: #0c0e17;
                border: 1px solid #1e2640;
                border-radius: 12px;
            }
            QPushButton {
                background-color: #161b2e;
                color: #ffffff;
                border: 1px solid #2a3454;
                border-radius: 8px;
                padding: 10px 18px;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.8px;
            }
            QPushButton:hover {
                background-color: #202844;
                border: 1px solid #00f0ff;
            }
            #listenBtn {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #059669, stop:1 #10b981);
                border: 1px solid #34d399;
            }
            #listenBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #10b981, stop:1 #059669);
            }
            #screenBtn {
                border: 1px solid #38bdf8;
            }
            #stopBtn {
                background-color: #3b0713;
                border: 1px solid #f43f5e;
                color: #fda4af;
            }
            #stopBtn:hover {
                background-color: #4c0519;
            }
        """)

    def add_message(self, text: str, sender: str = "user"):
        card = QFrame()
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 10, 14, 10)

        header_lbl = QLabel()
        header_lbl.setFont(QFont("Segoe UI", 9, QFont.Bold))

        body_lbl = QLabel(text)
        body_lbl.setWordWrap(True)
        body_lbl.setFont(QFont("Segoe UI", 11))

        if sender == "user":
            card.setStyleSheet("""
                background-color: #1a162b;
                border: 1px solid #3b2d54;
                border-left: 4px solid #a855f7;
                border-radius: 8px;
            """)
            header_lbl.setText("USER // VOICE INPUT")
            header_lbl.setStyleSheet("color: #c084fc;")
            body_lbl.setStyleSheet("color: #f1f5f9;")
        elif sender == "assistant":
            card.setStyleSheet("""
                background-color: #0b1e28;
                border: 1px solid #164e63;
                border-left: 4px solid #00f0ff;
                border-radius: 8px;
            """)
            header_lbl.setText("M.A.R.K. // AGENT RESPONSE")
            header_lbl.setStyleSheet("color: #00f0ff;")
            body_lbl.setStyleSheet("color: #ecfeff;")
        else:
            card.setStyleSheet("""
                background-color: #241804;
                border: 1px solid #78350f;
                border-left: 4px solid #fbbf24;
                border-radius: 8px;
            """)
            header_lbl.setText("AGENT ACTION // TOOL EXECUTION")
            header_lbl.setStyleSheet("color: #fbbf24;")
            body_lbl.setStyleSheet("color: #fef3c7; font-family: 'Consolas', monospace;")

        card_layout.addWidget(header_lbl)
        card_layout.addWidget(body_lbl)

        self.scroll_layout.insertWidget(self.scroll_layout.count() - 1, card)
        QTimer.singleShot(50, self.scroll_to_bottom)

    def scroll_to_bottom(self):
        sb = self.scroll_area.verticalScrollBar()
        sb.setValue(sb.maximum())

    def clear_chat(self):
        while self.scroll_layout.count() > 1:
            item = self.scroll_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def toggle_listening(self):
        if self.listener_thread and self.listener_thread.isRunning():
            self.stop_listening()
        else:
            self.start_listening()

    def start_listening(self):
        self.listener_thread = AgentListeningThread(self)
        self.listener_thread.status_changed.connect(self.update_status)
        self.listener_thread.tool_action_signaled.connect(self.on_tool_action)
        self.listener_thread.message_emitted.connect(self.add_message)
        self.listener_thread.speech_chunk_ready.connect(self.tts_worker.queue_sentence)
        self.listener_thread.latency_measured.connect(self.update_latency)
        self.listener_thread.start()

        self.listen_btn.setText("⏸️ PAUSE LISTENING")
        self.listen_btn.setStyleSheet("background-color: #854d0e; border: 1px solid #facc15;")

    def stop_listening(self):
        if self.listener_thread:
            self.listener_thread.stop()
            self.listener_thread.wait(500)
            self.listener_thread = None

        self.listen_btn.setText("🎙️ START LISTENING")
        self.listen_btn.setStyleSheet("")
        self.ai_core.set_state("IDLE")
        self.status_badge.setText("STANDBY // READY FOR VOICE COMMAND")

    def stop_all_actions(self):
        self.tts_worker.stop_speaking()
        self.action_banner.setText("⚠️ Emergency Halt Triggered")
        self.ai_core.set_state("IDLE")

    def inspect_screen_manual(self):
        self.update_status("INSPECTING SCREEN // GEMINI 2.5 FLASH", "THINKING")
        self.action_banner.setText("👁️ Tool: screen_vision(prompt='Full desktop overview')")
        QTimer.singleShot(100, self._do_screen_vision)

    def _do_screen_vision(self):
        res = MarkToolArsenal.screen_vision("Provide a sharp, 2-sentence summary of what is open on the user screen right now.")
        self.add_message(res, "assistant")
        self.tts_worker.queue_sentence(res)
        self.update_status("READY", "IDLE")

    def update_status(self, text: str, state: str):
        self.status_badge.setText(text.upper())
        self.ai_core.set_state(state)

    def on_tool_action(self, tool_name: str, args_str: str, result: str):
        self.action_banner.setText(f"🛠️ [EXECUTING: {tool_name}] -> {result[:60]}")
        self.add_message(f"🛠️ Tool: {tool_name}({args_str})\nResult: {result}", "tool")

    def update_latency(self, ttfa_ms: float):
        self.latency_label.setText(f"⚡ TTFA: {int(ttfa_ms)} ms")

    def on_speaking_started(self, text: str):
        self.ai_core.set_state("SPEAKING")
        self.status_badge.setText("M.A.R.K. SPEAKING // AUDIO STREAM ACTIVE")

    def on_speaking_finished(self):
        if self.listener_thread and self.listener_thread.isRunning():
            self.ai_core.set_state("LISTENING")
            self.status_badge.setText("LISTENING FOR NEXT COMMAND...")
        else:
            self.ai_core.set_state("IDLE")
            self.status_badge.setText("STANDBY // READY")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.stop_all_actions()
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self.stop_listening()
        self.tts_worker.shutdown()
        event.accept()


# ==============================================================================
# 5. HIGH-SPEED AGENT LISTENING THREAD (STREAMING PIPELINE)
# ==============================================================================

class AgentListeningThread(QThread):
    """
    Continuous voice listener + autonomous agent executor.
    Uses Groq Qwen 27B for sub-second intent extraction and tool execution.
    """
    status_changed = pyqtSignal(str, str)
    tool_action_signaled = pyqtSignal(str, str, str)
    message_emitted = pyqtSignal(str, str)
    speech_chunk_ready = pyqtSignal(str)
    latency_measured = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_running = True

    def stop(self):
        self.is_running = False

    def run(self):
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.8)
                if recognizer.energy_threshold < 150:
                    recognizer.energy_threshold = 150
                print(f"[Microphone] Calibrated energy threshold: {recognizer.energy_threshold}")

                while self.is_running:
                    self.status_changed.emit("Listening...", "LISTENING")
                    try:
                        audio = recognizer.listen(source, timeout=4, phrase_time_limit=15)
                    except sr.WaitTimeoutError:
                        continue

                    if not self.is_running:
                        break

                    t_start = time.time()
                    try:
                        query = recognizer.recognize_google(audio, language="en-IN")
                        self.message_emitted.emit(query, "user")
                        self.status_changed.emit("Agent Thinking & Routing...", "THINKING")

                        self.process_agent_query(query, t_start)

                    except sr.UnknownValueError:
                        pass
                    except sr.RequestError as e:
                        self.message_emitted.emit(f"Speech service error: {e}", "assistant")
                    except Exception as e:
                        print(f"[Listening Loop Error] {e}")

        except Exception as e:
            print(f"[Microphone Init Error] {e}")

    def process_agent_query(self, query: str, t_start: float):
        """Sends query to Groq Agent with Tool calling and streaming."""
        try:
            if not groq_client or not GROQ_API_KEY:
                self.message_emitted.emit("Groq API key is not configured. Please set GROQ_API_KEY in your environment or .env file.", "assistant")
                return

            completion = groq_client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=[
                    {"role": "system", "content": MarkAgentOrchestrator.SYSTEM_PROMPT},
                    {"role": "user", "content": query}
                ],
                max_tokens=220,
                temperature=0.1
            )

            raw_response = completion.choices[0].message.content.strip()

            tool_result_tuple = MarkAgentOrchestrator.parse_and_execute_tool(raw_response)

            final_response = ""
            if tool_result_tuple:
                tool_name, tool_args, tool_result = tool_result_tuple
                self.tool_action_signaled.emit(tool_name, tool_args, tool_result)

                voice_completion = groq_client.chat.completions.create(
                    model="qwen/qwen3.8-27b",
                    messages=[
                        {"role": "system", "content": "You are MARK. Provide a concise, friendly, 1-2 sentence spoken reply to the user based on the tool result."},
                        {"role": "user", "content": f"User Query: '{query}'\nTool Executed: {tool_name}\nTool Output: {tool_result}"}
                    ],
                    max_tokens=90,
                    temperature=0.1
                )
                final_response = voice_completion.choices[0].message.content.strip()
            else:
                final_response = raw_response

            ttfa = (time.time() - t_start) * 1000
            self.latency_measured.emit(ttfa)

            self.message_emitted.emit(final_response, "assistant")

            sentences = re.split(r'([.!?\n]+)', final_response)
            current_sent = ""
            for part in sentences:
                current_sent += part
                if any(p in part for p in ['.', '!', '?', '\n']) and len(current_sent.strip()) > 3:
                    self.speech_chunk_ready.emit(current_sent.strip())
                    current_sent = ""
            if current_sent.strip():
                self.speech_chunk_ready.emit(current_sent.strip())

        except Exception as e:
            print(f"[Agent Orchestration Error] {e}")
            self.message_emitted.emit(f"Agent processing error: {e}", "assistant")


# ==============================================================================
# 6. APPLICATION ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    hud = MarkCyberHUD()
    hud.show()
    sys.exit(app.exec_())
