"""
Quantum Error Correction IRC Bot Package

Components:
- qec_steane: Steane code QEC simulations with QuTiP
- midi_export: MIDI export for simulation data
- irc_bot: IRC bot implementation
- llm_integration: LLM-powered conversational AI
- integrated_bot: Full integration of all components
"""

__version__ = '1.0.0'

from .qec_steane import SteaneCode, ThresholdSimulation, SurfaceLattice
from .midi_export import MIDIConverter
from .irc_bot import IRCBot, QECIRCBot
from .llm_integration import LLMChatBot, MockLLMProvider
from .integrated_bot import IntegratedQECBot

__all__ = [
    'SteaneCode',
    'ThresholdSimulation',
    'SurfaceLattice',
    'MIDIConverter',
    'IRCBot',
    'QECIRCBot',
    'LLMChatBot',
    'MockLLMProvider',
    'IntegratedQECBot',
]
