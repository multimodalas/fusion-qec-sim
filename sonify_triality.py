#!/usr/bin/env python3
"""
E8 Triality Sonification → MIDI export
Maps logical error probabilities to three MIDI tracks:
Signal (+1), Loss (−2), Coherence (+1)
"""

import pandas as pd
import math
from mido import Message, MidiFile, MidiTrack, bpm2tempo

# Load data
df = pd.read_csv("qec_output.csv")

# MIDI setup
mid = MidiFile()
tempo = bpm2tempo(90)  # Φ = π/2 → 90 BPM baseline
ticks_per_beat = mid.ticks_per_beat
note_length = ticks_per_beat // 2

# Channel mapping
voices = {
    "Signal": {"mult": 1, "channel": 0},
    "Loss": {"mult": -2, "channel": 1},
    "Coherence": {"mult": 1, "channel": 2},
}

# Base pitch and scaling
base_pitch = 60  # Middle C
pitch_scale = 30  # how much pitch changes with log(error)

def prob_to_pitch(p):
    if p <= 0:
        return base_pitch
    return int(base_pitch + pitch_scale * -math.log10(p + 1e-12)) % 127

def prob_to_velocity(p):
    return int(max(20, min(120, 127 * (1 - math.log10(p + 1e-12) / 10))))

for voice_name, props in voices.items():
    track = MidiTrack()
    track.append(Message("program_change", program=0, channel=props["channel"], time=0))
    for _, row in df.iterrows():
        err, lx, lz, ly = row
        probs = [lx, lz, ly]
        avg_p = sum(probs) / 3
        pitch = prob_to_pitch(avg_p * abs(props["mult"]))
        vel = prob_to_velocity(err)
        # Note on / off pair
        track.append(Message("note_on", note=pitch, velocity=vel,
                             channel=props["channel"], time=0))
        track.append(Message("note_off", note=pitch, velocity=0,
                             channel=props["channel"], time=note_length))
    mid.tracks.append(track)

# Write output
mid.save("e8_triality_demo.mid")
print("💾 Wrote MIDI file: e8_triality_demo.mid (Signal/Loss/Coherence tracks)")
