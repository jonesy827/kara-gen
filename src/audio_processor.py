"""
Audio processing module for separating vocal and instrumental tracks using audio-separator.
"""

import os
import subprocess
import json
from typing import Dict, Optional, Tuple

from audio_separator.separator import Separator

class AudioProcessor:
    def __init__(self, output_dir: str):
        """Initialize the audio processor with output directory."""
        self.output_dir = str(output_dir)  # Ensure output_dir is a string
        self.separator = Separator()

    def trim_silence(self, input_file: str, output_dir: str) -> Tuple[str, float]:
        """
        Trim silence from the start of an audio file using FFmpeg.
        
        Args:
            input_file: Path to input audio file
            output_dir: Directory to save trimmed file
            
        Returns:
            Tuple of (trimmed_file_path, start_offset_seconds)
        """
        print("\nAnalyzing audio for silence...")
        
        # First, detect silence using FFmpeg silencedetect filter
        silence_cmd = [
            'ffmpeg', '-i', input_file,
            '-af', 'silencedetect=noise=-30dB:d=0.5',
            '-f', 'null', '-'
        ]
        
        try:
            result = subprocess.run(
                silence_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse FFmpeg output to find silence end
            silence_end = 0.0
            for line in result.stderr.split('\n'):
                if 'silence_end' in line:
                    try:
                        silence_end = float(line.split('silence_end: ')[1].split(' ')[0])
                        break  # Use first silence end (end of initial silence)
                    except (IndexError, ValueError):
                        continue
            
            if silence_end > 0:
                print(f"Found {silence_end:.2f} seconds of silence at start")
                
                # Create trimmed output file
                input_base = os.path.splitext(os.path.basename(input_file))[0]
                trimmed_file = os.path.join(output_dir, f"{input_base}_trimmed.wav")
                
                # Trim the silence using FFmpeg
                trim_cmd = [
                    'ffmpeg', '-y',  # Overwrite output file if exists
                    '-ss', str(silence_end),  # Start time
                    '-i', input_file,
                    '-acodec', 'pcm_s16le',  # Use WAV format
                    trimmed_file
                ]
                
                subprocess.run(trim_cmd, check=True, capture_output=True)
                print(f"Trimmed {silence_end:.2f} seconds from start of audio")
                return trimmed_file, silence_end
            
            print("No significant silence detected at start of file")
            return input_file, 0.0
            
        except subprocess.CalledProcessError as e:
            print(f"Error during silence detection: {e}")
            print("Continuing with original file")
            return input_file, 0.0

    def process_audio(self, input_file: str, artist: str, track: str) -> Dict[str, str]:
        """
        Process audio file to generate karaoke tracks:
        1. First pass: Separate vocals and instrumental using MDX23C-InstVoc
        2. Second pass: Process vocals to separate lead and backing vocals using 5_HP-Karaoke-UVR
        
        Returns:
            Dict containing paths to generated audio files and timing info
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
        
        print(f"Expected first pass outputs:")
        print(f"- Vocals: {vocals_file}")
        print(f"- Instrumental: {instrumental_file}")
        
        # Check if first pass files already exist
        if os.path.exists(vocals_file) and os.path.exists(instrumental_file):
            print("First pass files already exist, skipping first pass separation")
            # Verify file sizes
            vocals_size = os.path.getsize(vocals_file)
            instrumental_size = os.path.getsize(instrumental_file)
            print(f"Existing file sizes:")
            print(f"- Vocals: {vocals_size / 1024 / 1024:.2f} MB")
            print(f"- Instrumental: {instrumental_size / 1024 / 1024:.2f} MB")
        else:
            print("Starting first pass separation...")
            try:
                # Create new separator instance with first pass settings
                self.separator = Separator(
                    output_dir=first_pass_output,
                    output_format="wav"
                )
                # Load and apply the model
                print("Loading MDX23C model...")
                self.separator.load_model("MDX23C-8KFFT-InstVoc_HQ.ckpt")
                print("Separating audio...")
                self.separator.separate(str(input_file))
                
                # Verify the separation worked
                if not os.path.exists(vocals_file):
                    raise Exception(f"Vocals file not created: {vocals_file}")
                if not os.path.exists(instrumental_file):
                    raise Exception(f"Instrumental file not created: {instrumental_file}")
                    
                # Check file sizes
                vocals_size = os.path.getsize(vocals_file)
                instrumental_size = os.path.getsize(instrumental_file)
                print(f"Generated file sizes:")
                print(f"- Vocals: {vocals_size / 1024 / 1024:.2f} MB")
                print(f"- Instrumental: {instrumental_size / 1024 / 1024:.2f} MB")
                
                if vocals_size == 0 or instrumental_size == 0:
                    raise Exception("One or both output files are empty")
                    
            except Exception as e:
                print(f"Error during first pass separation: {str(e)}")
                print("Files in output directory:")
                for f in os.listdir(first_pass_output):
                    fpath = os.path.join(first_pass_output, f)
                    fsize = os.path.getsize(fpath)
                    print(f"- {f}: {fsize / 1024 / 1024:.2f} MB")
                raise

        vocals_file = str(vocals_file)  # Ensure vocals_file is a string
        print(f"Vocals file path: {vocals_file}")
        
        # Trim silence from vocals before second pass
        trimmed_vocals, start_offset = self.trim_silence(vocals_file, first_pass_output)
        
        # Second pass: Process vocals to separate lead and backing
        second_pass_output = os.path.join(song_dir, "second_pass")
        second_pass_output = str(second_pass_output)  # Ensure second_pass_output is a string
        os.makedirs(second_pass_output, exist_ok=True)
        print(f"Second pass output directory: {second_pass_output}")
        
        # Get expected second pass output file paths
        vocals_base = os.path.splitext(os.path.basename(trimmed_vocals))[0]
        second_pass_vocals = os.path.join(second_pass_output, f"{vocals_base}_(Vocals)_5_HP-Karaoke-UVR.wav")
        second_pass_instrumental = os.path.join(second_pass_output, f"{vocals_base}_(Instrumental)_5_HP-Karaoke-UVR.wav")
        
        print(f"Expected second pass outputs:")
        print(f"- Lead vocals: {second_pass_vocals}")
        print(f"- Backing vocals: {second_pass_instrumental}")
        
        # Check if second pass files already exist
        if os.path.exists(second_pass_vocals) and os.path.exists(second_pass_instrumental):
            print("Second pass files already exist, skipping second pass separation")
            # Verify file sizes
            vocals_size = os.path.getsize(second_pass_vocals)
            instrumental_size = os.path.getsize(second_pass_instrumental)
            print(f"Existing file sizes:")
            print(f"- Lead vocals: {vocals_size / 1024 / 1024:.2f} MB")
            print(f"- Backing vocals: {instrumental_size / 1024 / 1024:.2f} MB")
        else:
            print("Starting second pass separation...")
            try:
                # Create new separator instance with second pass settings
                self.separator = Separator(
                    output_dir=second_pass_output,
                    output_format="wav"
                )
                # Load and apply the model
                print("Loading 5_HP-Karaoke model...")
                self.separator.load_model("5_HP-Karaoke-UVR.pth")
                print("Separating vocals...")
                self.separator.separate(trimmed_vocals)
                
                # Verify the separation worked
                if not os.path.exists(second_pass_vocals):
                    raise Exception(f"Lead vocals file not created: {second_pass_vocals}")
                if not os.path.exists(second_pass_instrumental):
                    raise Exception(f"Backing vocals file not created: {second_pass_instrumental}")
                    
                # Check file sizes
                vocals_size = os.path.getsize(second_pass_vocals)
                instrumental_size = os.path.getsize(second_pass_instrumental)
                print(f"Generated file sizes:")
                print(f"- Lead vocals: {vocals_size / 1024 / 1024:.2f} MB")
                print(f"- Backing vocals: {instrumental_size / 1024 / 1024:.2f} MB")
                
                if vocals_size == 0 or instrumental_size == 0:
                    raise Exception("One or both output files are empty")
                    
            except Exception as e:
                print(f"Error during second pass separation: {str(e)}")
                print("Files in output directory:")
                for f in os.listdir(second_pass_output):
                    fpath = os.path.join(second_pass_output, f)
                    fsize = os.path.getsize(fpath)
                    print(f"- {f}: {fsize / 1024 / 1024:.2f} MB")
                raise

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

        # Save timing info
        timing_info = {
            "start_offset": start_offset,
            "trimmed_vocals": os.path.basename(trimmed_vocals) if trimmed_vocals != vocals_file else None
        }
        timing_file = os.path.join(song_dir, "timing_info.json")
        with open(timing_file, 'w') as f:
            json.dump(timing_info, f, indent=2)
        output_files["timing_info"] = timing_file

        # Clean up temporary directories if they're empty
        try:
            os.rmdir(first_pass_output)
            os.rmdir(second_pass_output)
        except OSError:
            # Ignore errors if directories aren't empty or are already gone
            pass

        return output_files 