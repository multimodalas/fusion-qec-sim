"""
MIDI Export Module for QEC Simulations

Converts quantum error correction simulation data to MIDI:
- Physical error rates → tempo (0.01 = 120 BPM)
- Logical dips → e-minor arpeggios
- Eigenvalues → velocities [8, 92]
- Notes forming arpeggios C3-G5 on helix flip
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
try:
    from mido import Message, MidiFile, MidiTrack, MetaMessage
except ImportError:
    MidiFile = None
    print("Warning: mido not installed. MIDI export will be limited.")


class MIDIConverter:
    """
    Convert QEC simulation data to MIDI format.
    """
    
    # Note mapping for e-minor arpeggio: E-G-B
    E_MINOR_NOTES = {
        'E3': 52,  # E3
        'G3': 55,  # G3
        'B3': 59,  # B3
        'E4': 64,  # E4
        'G4': 67,  # G4
        'B4': 71,  # B4
        'E5': 76,  # E5
    }
    
    # Extended note range C3-G5
    NOTE_RANGE = {
        'C3': 48, 'D3': 50, 'E3': 52, 'F3': 53, 'G3': 55, 'A3': 57, 'B3': 59,
        'C4': 60, 'D4': 62, 'E4': 64, 'F4': 65, 'G4': 67, 'A4': 69, 'B4': 71,
        'C5': 72, 'D5': 74, 'E5': 76, 'F5': 77, 'G5': 79,
    }
    
    def __init__(self):
        """Initialize MIDI converter."""
        self.ticks_per_beat = 480
        self.min_velocity = 8
        self.max_velocity = 92
        
    def error_rate_to_tempo(self, error_rate: float) -> int:
        """
        Convert physical error rate to BPM tempo.
        
        Args:
            error_rate: Physical error rate (e.g., 0.01)
            
        Returns:
            Tempo in BPM (capped between 60 and 240)
        """
        # Map 0.01 → 120 BPM linearly
        # error_rate * 12000 = BPM
        tempo = int(error_rate * 12000)
        
        # Cap tempo to reasonable range
        tempo = max(60, min(240, tempo))
        
        return tempo
    
    def eigenvalue_to_velocity(self, eigenvalue: float) -> int:
        """
        Map eigenvalue to MIDI velocity [8, 92].
        
        Args:
            eigenvalue: Eigenvalue in range [-1, 1]
            
        Returns:
            MIDI velocity (8-92)
        """
        # Normalize eigenvalue from [-1, 1] to [0, 1]
        normalized = (eigenvalue + 1) / 2
        
        # Map to velocity range
        velocity = int(self.min_velocity + normalized * 
                      (self.max_velocity - self.min_velocity))
        
        return max(self.min_velocity, min(self.max_velocity, velocity))
    
    def logical_error_to_arpeggio(
        self,
        logical_error_rate: float,
        duration: int = 480
    ) -> List[Tuple[int, int, int]]:
        """
        Convert logical error rate to e-minor arpeggio pattern.
        
        Args:
            logical_error_rate: Logical error rate
            duration: Duration per note in ticks
            
        Returns:
            List of (note, velocity, duration) tuples
        """
        arpeggio = []
        
        # Choose arpeggio notes based on error magnitude
        if logical_error_rate < 0.01:
            # Low error: ascending arpeggio
            notes = ['E3', 'G3', 'B3', 'E4']
        elif logical_error_rate < 0.1:
            # Medium error: full arpeggio
            notes = ['E3', 'G3', 'B3', 'E4', 'G4']
        else:
            # High error: extended arpeggio
            notes = ['E3', 'G3', 'B3', 'E4', 'G4', 'B4', 'E5']
            
        # Convert to MIDI notes with varying velocity
        base_velocity = int(50 + logical_error_rate * 100)
        base_velocity = max(20, min(100, base_velocity))
        
        for i, note_name in enumerate(notes):
            note = self.E_MINOR_NOTES[note_name]
            velocity = base_velocity + (i * 5)  # Slight increase per note
            velocity = min(127, velocity)
            arpeggio.append((note, velocity, duration))
            
        return arpeggio
    
    def threshold_curve_to_midi(
        self,
        p_phys: np.ndarray,
        p_log: np.ndarray,
        output_path: str
    ) -> Optional[MidiFile]:
        """
        Convert threshold curve data to MIDI file.
        
        Args:
            p_phys: Physical error rates
            p_log: Logical error rates
            output_path: Output MIDI file path
            
        Returns:
            MidiFile object if mido available, None otherwise
        """
        if MidiFile is None:
            print("Error: mido library not available. Cannot create MIDI file.")
            print("Install with: pip install mido")
            return None
            
        # Create MIDI file
        mid = MidiFile(ticks_per_beat=self.ticks_per_beat)
        track = MidiTrack()
        mid.tracks.append(track)
        
        # Add track name
        track.append(MetaMessage('track_name', name='QEC Threshold Curve', time=0))
        
        # Initial tempo
        initial_tempo = self.error_rate_to_tempo(p_phys[0])
        track.append(MetaMessage('set_tempo', tempo=int(60000000 / initial_tempo), time=0))
        
        current_time = 0
        
        # Convert each data point to MIDI events
        for i, (phys_rate, log_rate) in enumerate(zip(p_phys, p_log)):
            # Update tempo based on physical error rate
            tempo_bpm = self.error_rate_to_tempo(phys_rate)
            tempo_microseconds = int(60000000 / tempo_bpm)
            track.append(MetaMessage('set_tempo', tempo=tempo_microseconds, time=current_time))
            
            # Create arpeggio for logical error
            arpeggio = self.logical_error_to_arpeggio(log_rate)
            
            # Add arpeggio notes
            for note, velocity, duration in arpeggio:
                # Note on
                track.append(Message('note_on', note=note, velocity=velocity, 
                                    time=0, channel=0))
                # Note off
                track.append(Message('note_off', note=note, velocity=0, 
                                    time=duration, channel=0))
                
            current_time = 240  # Time between data points
            
        # Save MIDI file
        mid.save(output_path)
        print(f"MIDI file saved to: {output_path}")
        
        return mid
    
    def eigenvalues_to_melody(
        self,
        eigenvalues: Dict[str, float],
        output_path: str,
        note_duration: int = 480
    ) -> Optional[MidiFile]:
        """
        Convert eigenvalue spectrum to melodic sequence.
        
        Args:
            eigenvalues: Dictionary of eigenvalues
            output_path: Output MIDI file path
            note_duration: Duration per note in ticks
            
        Returns:
            MidiFile object if mido available, None otherwise
        """
        if MidiFile is None:
            print("Error: mido library not available. Cannot create MIDI file.")
            return None
            
        # Create MIDI file
        mid = MidiFile(ticks_per_beat=self.ticks_per_beat)
        track = MidiTrack()
        mid.tracks.append(track)
        
        # Add track name
        track.append(MetaMessage('track_name', name='QEC Eigenvalue Spectrum', time=0))
        
        # Set tempo
        track.append(MetaMessage('set_tempo', tempo=500000, time=0))  # 120 BPM
        
        # Get note sequence
        note_names = list(self.NOTE_RANGE.keys())
        
        # Convert eigenvalues to notes
        for i, (key, eigenvalue) in enumerate(eigenvalues.items()):
            # Map to note
            note_idx = i % len(note_names)
            note = self.NOTE_RANGE[note_names[note_idx]]
            
            # Map eigenvalue to velocity
            velocity = self.eigenvalue_to_velocity(eigenvalue)
            
            # Add note
            track.append(Message('note_on', note=note, velocity=velocity, 
                                time=0, channel=0))
            track.append(Message('note_off', note=note, velocity=0, 
                                time=note_duration, channel=0))
            
        # Save MIDI file
        mid.save(output_path)
        print(f"MIDI file saved to: {output_path}")
        
        return mid
    
    def syndrome_pattern_to_percussion(
        self,
        syndrome_pattern: np.ndarray,
        output_path: str
    ) -> Optional[MidiFile]:
        """
        Convert syndrome pattern to percussion MIDI.
        
        Args:
            syndrome_pattern: 2D array of syndromes (0 or 1)
            output_path: Output MIDI file path
            
        Returns:
            MidiFile object if mido available, None otherwise
        """
        if MidiFile is None:
            print("Error: mido library not available. Cannot create MIDI file.")
            return None
            
        # Create MIDI file
        mid = MidiFile(ticks_per_beat=self.ticks_per_beat)
        track = MidiTrack()
        mid.tracks.append(track)
        
        # Add track name
        track.append(MetaMessage('track_name', name='QEC Syndrome Pattern', time=0))
        
        # Set tempo
        track.append(MetaMessage('set_tempo', tempo=500000, time=0))  # 120 BPM
        
        # Percussion notes (MIDI channel 9, notes 35-81)
        kick_drum = 36
        snare_drum = 38
        hi_hat = 42
        
        # Convert syndrome pattern to percussion
        flat_pattern = syndrome_pattern.flatten()
        
        for i, syndrome in enumerate(flat_pattern):
            if syndrome == 1:
                # Syndrome detected: play snare
                note = snare_drum
                velocity = 80
            else:
                # No syndrome: play hi-hat
                note = hi_hat
                velocity = 40
                
            # Add note on channel 9 (percussion)
            track.append(Message('note_on', note=note, velocity=velocity, 
                                time=0, channel=9))
            track.append(Message('note_off', note=note, velocity=0, 
                                time=240, channel=9))
            
        # Save MIDI file
        mid.save(output_path)
        print(f"MIDI file saved to: {output_path}")
        
        return mid


def demo_midi_export():
    """
    Demonstrate MIDI export functionality.
    """
    print("=== MIDI Export Demo ===\n")
    
    converter = MIDIConverter()
    
    # Demo 1: Error rate to tempo
    print("Demo 1: Error Rate to Tempo")
    test_rates = [0.001, 0.01, 0.05, 0.1]
    for rate in test_rates:
        tempo = converter.error_rate_to_tempo(rate)
        print(f"  Error rate {rate:.3f} → {tempo} BPM")
    
    # Demo 2: Eigenvalue to velocity
    print("\nDemo 2: Eigenvalue to Velocity")
    test_eigenvalues = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for eig in test_eigenvalues:
        velocity = converter.eigenvalue_to_velocity(eig)
        print(f"  Eigenvalue {eig:+.1f} → velocity {velocity}")
    
    # Demo 3: Logical error to arpeggio
    print("\nDemo 3: Logical Error to Arpeggio")
    test_log_rates = [0.001, 0.05, 0.2]
    for log_rate in test_log_rates:
        arpeggio = converter.logical_error_to_arpeggio(log_rate)
        print(f"  Logical error {log_rate:.3f} → {len(arpeggio)} notes")
    
    # Demo 4: Create sample MIDI file (if mido available)
    if MidiFile is not None:
        print("\nDemo 4: Creating Sample MIDI Files")
        
        # Sample threshold curve
        p_phys = np.logspace(-4, -1, 5)
        p_log = p_phys ** 2  # Simplified logical error
        
        output_path = '/tmp/qec_threshold_curve.mid'
        converter.threshold_curve_to_midi(p_phys, p_log, output_path)
        
        # Sample eigenvalues
        eigenvalues = {f'X_{i}': np.random.uniform(-1, 1) for i in range(7)}
        output_path = '/tmp/qec_eigenvalues.mid'
        converter.eigenvalues_to_melody(eigenvalues, output_path)
        
        # Sample syndrome pattern
        syndrome_pattern = np.random.randint(0, 2, size=(5, 5))
        output_path = '/tmp/qec_syndromes.mid'
        converter.syndrome_pattern_to_percussion(syndrome_pattern, output_path)
    else:
        print("\nDemo 4: MIDI file creation skipped (mido not installed)")
    
    print("\n=== Demo Complete ===")


if __name__ == '__main__':
    demo_midi_export()
