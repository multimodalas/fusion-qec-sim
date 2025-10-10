# IRC Bot Guide

## Overview

The fusion-qec-sim IRC bot is an AI-powered assistant that integrates quantum error correction simulations with IRC chat capabilities. It combines:

- **QuTiP-based Steane Code Simulations**: Full [[7,1,3]] code implementation with depolarizing noise
- **MIDI Export**: Convert simulation data to musical representations
- **LLM Integration**: Conversational AI for code generation and explanation
- **IRC Protocol**: Real-time communication on IRC servers

## Quick Start

```bash
# Demo mode (no IRC connection)
python run_bot.py --demo

# Connect to IRC server
export IRC_SERVER=irc.libera.chat
export IRC_CHANNEL=#qec-sim
python run_bot.py
```

## Available Commands

- `!help` - Display available commands
- `!runsim [error_rate]` - Run Steane code simulation
- `!threshold` - Display pseudo-threshold
- `!ai <question>` - Ask AI about QEC
- `!note <note>` - Play MIDI note
- `!export [type]` - Export to MIDI

See full documentation in the repository for detailed usage.
