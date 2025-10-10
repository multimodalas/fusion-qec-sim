"""
Complete QEC Simulation Demo

Demonstrates all features of the fusion-qec-sim IRC bot components.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qec_steane import SteaneCode, ThresholdSimulation
from midi_export import MIDIConverter
from llm_integration import LLMChatBot, MockLLMProvider
import numpy as np


def main():
    """Run complete demo."""
    print("\n" + "=" * 60)
    print("FUSION-QEC-SIM: Complete Demo")
    print("=" * 60)
    
    # Demo 1: Steane Code
    print("\n1. Steane Code Simulation")
    code = SteaneCode()
    state = code.encode_logical_zero()
    noisy = code.apply_depolarizing_noise(state, 0.01)
    print(f"   ✓ Encoded and applied noise")
    
    # Demo 2: MIDI Export
    print("\n2. MIDI Export")
    converter = MIDIConverter()
    p_phys = np.logspace(-3, -1, 5)
    p_log = p_phys ** 2
    converter.threshold_curve_to_midi(p_phys, p_log, '/tmp/qec_demo.mid')
    print(f"   ✓ Exported to MIDI")
    
    # Demo 3: LLM Integration
    print("\n3. LLM Integration")
    bot = LLMChatBot(MockLLMProvider())
    response = bot.generate_response("What is QEC?", user="demo")
    print(f"   ✓ AI response generated")
    
    print("\n" + "=" * 60)
    print("Demo complete! Check /tmp/qec_demo.mid")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
