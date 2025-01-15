"""
Audio processing module for separating vocal and instrumental tracks using audio-separator.
"""

import os
from typing import Dict, Optional
from audio_separator.separator import Separator

class AudioProcessor:
    def __init__(self, output_dir: str):
        """Initialize the audio processor with output directory."""
        self.output_dir = str(output_dir)  # Ensure output_dir is a string
        self.separator = Separator()

    def process_audio(self, input_file: str, artist: str, track: str) -> Dict[str, str]:
        """
        Process audio file to generate karaoke tracks:
        1. First pass: Separate vocals and instrumental using MDX23C-InstVoc
        2. Second pass: Process vocals to separate lead and backing vocals using 5_HP-Karaoke-UVR
        
        Returns:
            Dict containing paths to generated audio files
        """
        print(f"Input file: {input_file}")
        print(f"Artist: {artist}")
        print(f"Track: {track}")
        
        # Create song-specific output directory
        song_dir = os.path.join(self.output_dir, f"{artist} - {track}")
        song_dir = str(song_dir)  # Ensure song_dir is a string
        os.makedirs(song_dir, exist_ok=True)
        print(f"Song directory: {song_dir}")

        # First pass: Separate vocals and instrumental
        first_pass_output = os.path.join(song_dir, "first_pass")
        first_pass_output = str(first_pass_output)  # Ensure first_pass_output is a string
        os.makedirs(first_pass_output, exist_ok=True)
        print(f"First pass output directory: {first_pass_output}")
        
        # Get expected output file paths
        input_base = os.path.splitext(os.path.basename(input_file))[0]
        vocals_file = os.path.join(first_pass_output, f"{input_base}_(Vocals)_MDX23C-8KFFT-InstVoc_HQ.wav")
        instrumental_file = os.path.join(first_pass_output, f"{input_base}_(Instrumental)_MDX23C-8KFFT-InstVoc_HQ.wav")
        
        # Check if first pass files already exist
        if os.path.exists(vocals_file) and os.path.exists(instrumental_file):
            print("First pass files already exist, skipping first pass separation")
        else:
            print("Starting first pass separation...")
            # Create new separator instance with first pass settings
            self.separator = Separator(
                output_dir=first_pass_output,
                output_format="wav"
            )
            # Load and apply the model
            self.separator.load_model("MDX23C-8KFFT-InstVoc_HQ.ckpt")
            self.separator.separate(str(input_file))

        vocals_file = str(vocals_file)  # Ensure vocals_file is a string
        print(f"Vocals file path: {vocals_file}")
        
        # Second pass: Process vocals to separate lead and backing
        second_pass_output = os.path.join(song_dir, "second_pass")
        second_pass_output = str(second_pass_output)  # Ensure second_pass_output is a string
        os.makedirs(second_pass_output, exist_ok=True)
        print(f"Second pass output directory: {second_pass_output}")
        
        # Get expected second pass output file paths
        vocals_base = os.path.splitext(os.path.basename(vocals_file))[0]
        second_pass_vocals = os.path.join(second_pass_output, f"{vocals_base}_(Vocals)_5_HP-Karaoke-UVR.wav")
        second_pass_instrumental = os.path.join(second_pass_output, f"{vocals_base}_(Instrumental)_5_HP-Karaoke-UVR.wav")
        
        # Check if second pass files already exist
        if os.path.exists(second_pass_vocals) and os.path.exists(second_pass_instrumental):
            print("Second pass files already exist, skipping second pass separation")
        else:
            print("Starting second pass separation...")
            # Create new separator instance with second pass settings
            self.separator = Separator(
                output_dir=second_pass_output,
                output_format="wav"
            )
            # Load and apply the model
            self.separator.load_model("5_HP-Karaoke-UVR.pth")
            self.separator.separate(vocals_file)

        # Move and rename final files
        output_files = {}
        
        # No vocals track (instrumental from first pass)
        no_vocals = os.path.join(song_dir, "no_vocals.wav")
        no_vocals = str(no_vocals)  # Ensure no_vocals is a string
        if not os.path.exists(no_vocals):
            os.rename(instrumental_file, no_vocals)
        output_files["no_vocals"] = no_vocals

        # Lead vocals (vocals from second pass)
        lead_vocals = os.path.join(song_dir, "lead_vocals.wav")
        lead_vocals = str(lead_vocals)  # Ensure lead_vocals is a string
        if not os.path.exists(lead_vocals):
            os.rename(second_pass_vocals, lead_vocals)
        output_files["lead_vocals"] = lead_vocals

        # Backing vocals (instrumental from second pass)
        backing_vocals = os.path.join(song_dir, "backing_vocals.wav")
        backing_vocals = str(backing_vocals)  # Ensure backing_vocals is a string
        if not os.path.exists(backing_vocals):
            os.rename(second_pass_instrumental, backing_vocals)
        output_files["backing_vocals"] = backing_vocals

        # Clean up temporary directories if they're empty
        try:
            os.rmdir(first_pass_output)
            os.rmdir(second_pass_output)
        except OSError:
            # Ignore errors if directories aren't empty or are already gone
            pass

        return output_files 