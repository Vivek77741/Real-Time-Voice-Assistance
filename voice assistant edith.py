"""
Autonomous Multimodal LLM Voice & Computer-Use Agent (E.D.I.T.H. / M.A.R.K.)
=============================================================================
This file serves as the seamless primary entrypoint for the Real-Time Voice Assistance repository.
It launches the high-performance autonomous agent with PyQt5 Cyber HUD, Groq Qwen 27B reasoning,
Gemini 2.5 Flash screen vision, PyAutoGUI laptop control, and sub-second streaming TTS.
"""

import sys
import os

# Ensure current directory is on python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mark_voice_assistant import MarkCyberHUD, QApplication

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    hud = MarkCyberHUD()
    hud.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
