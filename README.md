[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/multimodalas/fusion-qec-sim/blob/main/notebooks/qec_demo_global.ipynb)

# fusion-qec-sim

Creative quantum error correction, DNA analysis, 3D cube visualization, and **AI-powered IRC bot**—open, modular, and minimal.

## New: AI-Powered IRC Bot

The fusion-qec-sim project now includes a fully-featured IRC bot that combines:

- **QuTiP-based Steane Code Simulations**: [[7,1,3]] code with depolarizing noise, pseudo-threshold calculations (η_thr ≈ 9.3×10^{-5})
- **MIDI Export**: Convert simulation data to music (error rates → tempo, eigenvalues → velocities, logical errors → e-minor arpeggios)
- **LLM Integration**: Conversational AI for code generation, simulation explanation, and chat moderation
- **IRC Protocol**: Real-time Q&A and code review in IRC channels

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run in demo mode (no IRC connection)
python run_bot.py --demo

# Connect to IRC server
export IRC_SERVER=irc.libera.chat
export IRC_CHANNEL=#qec-sim
python run_bot.py
```

### Available Commands

- `!runsim [error_rate]` - Run Steane code simulation
- `!threshold` - Display pseudo-threshold
- `!ai <question>` - Ask AI about QEC concepts
- `!export [type]` - Export simulation to MIDI
- `!note <note>` - Play MIDI note (C3-G5)

See [docs/IRC_BOT_GUIDE.md](docs/IRC_BOT_GUIDE.md) for complete documentation.

### Examples

```bash
# Run complete demo
python examples/qec_demo_full.py

# Run individual modules
python src/qec_steane.py        # QEC simulations
python src/midi_export.py       # MIDI export
python src/llm_integration.py   # LLM features
```

---

## Original Features

...