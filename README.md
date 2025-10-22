[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/multimodalas/fusion-qec-sim/blob/main/notebooks/qec_demo_global.ipynb)
Quantum Error Correction — fusion-qec-sim

Creative quantum error correction, DNA analysis, 3D lattice visualization, and AI-powered IRC orchestration.

⚛ Conceptual Basis

## New: AI-Powered IRC Bot with Dual Backend Support

<pre> Φ = π / 2  SCL DIAG + [1, −2, 1] </pre>

- **Dual Backend Support**: Choose between QuTiP or Qiskit for quantum simulations
- **Steane Code [[7,1,3]]**: Depolarizing noise, pseudo-threshold calculations (η_thr ≈ 9.3×10^{-5})
- **MIDI Export**: Convert simulation data to music (error rates → tempo, eigenvalues → velocities, logical errors → e-minor arpeggios)
- **LLM Integration**: Conversational AI for code generation, simulation explanation, and chat moderation
- **IRC Protocol**: Real-time Q&A and code review in IRC channels

In practical terms:

```bash
# Install dependencies (includes both QuTiP and Qiskit)
pip install -r requirements.txt

Information Entropy → Signal Mapping — translating quantum noise dynamics into musical structure.

# Connect to IRC server with QuTiP backend (default)
export IRC_SERVER=irc.libera.chat
export IRC_CHANNEL=#qec-sim
python run_bot.py

# Or use Qiskit backend
export QEC_BACKEND=qiskit
python run_bot.py
```

💬 AI-Powered IRC Bot

- `!runsim [error_rate]` - Run Steane code simulation
- `!threshold` - Display pseudo-threshold
- `!backend [qutip|qiskit]` - Switch quantum backend
- `!ai <question>` - Ask AI about QEC concepts
- `!export [type]` - Export simulation to MIDI
- `!note <note>` - Play MIDI note (C3-G5)

See [docs/IRC_BOT_GUIDE.md](docs/IRC_BOT_GUIDE.md) for complete documentation.

### Examples

```bash
# Run complete demo
python examples/qec_demo_full.py

# Run individual modules
python src/qec_steane.py        # QEC simulations (QuTiP)
python src/qec_steane.py --compare  # Compare QuTiP vs Qiskit
python src/midi_export.py       # MIDI export
python src/llm_integration.py   # LLM features
```

---

## Original Features

...
