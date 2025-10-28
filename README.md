[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/multimodalas/fusion-qec-sim/blob/main/notebooks/qec_demo_global.ipynb)

# Quantum Error Correction — QEC / Fusion-QEC-Sim

Creative quantum error correction, DNA analysis, 3D lattice visualization, and AI-powered IRC orchestration.

---

## ⚛ Conceptual Basis

This project builds on the **E8 Triality Framework** — uniting symmetry, information, and computation — under the guiding relation:

<pre>Φ = π / 2  SCL DIAG + [ 1 , −2 , 1 ]</pre>

This defines a ternary balance across the quantum–classical boundary,  
where each component of the `[ 1, −2, 1 ]` vector encodes a reversible polarity  
between **signal**, **coherence**, and **loss**.

### In practical terms

- **Fusion-QEC (Photonic)** — modular, loss-tolerant error correction inspired by MBQC fusion gates.  
- **Information Entropy → Signal Mapping** — translating quantum noise dynamics into musical structure.  
- **Triality Framework** — treating computation, geometry, and perception as three projections of a single invariant form.

---

## 💬 AI-Powered IRC Bot

An integrated assistant linking **simulation**, **music**, and **conversation**:

| Feature | Description |
|----------|-------------|
| **QuTiP-based Steane Code** | [[7, 1, 3]] simulation with depolarizing noise and pseudo-threshold (ηₜₕᵣ ≈ 9.3 × 10⁻⁵) |
| **MIDI Export** | Converts error metrics → tempo, eigenvalues → velocity, logical errors → E-minor arpeggios |
| **LLM Integration** | Conversational AI for code generation, simulation commentary, and live moderation |
| **IRC Protocol** | Real-time Q&A, simulation control, and generative music triggers |

---

## 🎧 E8 Triality Sonification Demo

Experience the sound of **symmetry meeting error correction**.  
This audio was generated directly from QEC simulation data using the  
<code>Φ = π / 2 SCL DIAG + [ 1 , − 2 , 1 ]</code> mapping.

> Each amplitude and interval reflects the balance between signal, coherence, and loss —  
> a musical rendering of the **Fusion-QEC Triality Model**.

<audio controls>
  <source src="e8_triality.wav" type="audio/wav">
  Your browser does not support the audio element.  
  [Download the demo](./e8_triality.wav)
</audio>

---

## 📁 Repository Assets

- 🖼 **Figure 1** — [View / Download](./Figure_1.png)  
- 🖼 **Figure 2** — [View / Download](./Figure_2.png)  
- 🎵 **Sonification MP3** — [Listen: QEC Fault Lines Sonification](./QEC%20Fault%20Lines%20Sonification.mp3)  
- 📄 **Benchmark Report (PDF)** — [Read: QEC Benchmark Report](./QEC_Benchmark_Report.pdf)  
- 📦 **Full Repo Archive (ZIP)** — [Download: QEC_Repo.zip](./QEC_Repo.zip)  
- 💻 **Sonification Script** — [View Source: sonify_triality.py](./sonify_triality.py)

> All resources are located in the repository root for direct access and reproducibility.

---

## 🧠 At a Glance

- **Language:** Python 3.11+  
- **Core Dependencies:** NumPy, Pandas, QuTiP, Plotly, Mido  
- **Environment:** Arch Linux / PipeWire audio  
- **Output:** Live audio or export to WAV (`e8_triality.wav`)  
- **Data Input:** `qec_output.csv` or `qec_data_prepared.csv`

---

# Connect to IRC server with QuTiP backend (default)
export IRC_SERVER=irc.libera.chat
export IRC_CHANNEL=#qec-sim
python run_bot.py

# Or use Qiskit backend
export QEC_BACKEND=qiskit
python run_bot.py
```

```bash
# Install dependencies
pip install -r requirements.txt

# Generate sonification (auto-fallback to WAV)
python sonify_triality.py
For more complex experiments, see QEC_Benchmark_Report.pdf
or remix the exported audio in your preferred DAW.

© 2025 QSOLKCB / Trent Slade. All rights reserved.
Open collaboration welcome under MIT License.
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
