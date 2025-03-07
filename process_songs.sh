#!/bin/bash

# Activate virtual environment
source venv_py310/bin/activate

# Create finished directory if it doesn't exist
mkdir -p finished

# Process each .flac file in the ready directory
for song in ready/*.flac; do
    if [ -f "$song" ]; then
        # Extract filename without path and extension
        filename=$(basename "$song" .flac)
        
        # Remove track number prefix (XX - )
        filename_no_track=${filename#*- }
        
        # Split into artist and title
        artist=$(echo "$filename_no_track" | cut -d '-' -f1 | sed 's/[[:space:]]*$//')
        title_with_extra=$(echo "$filename_no_track" | cut -d '-' -f2- | sed 's/^[[:space:]]*//')
        
        # Remove parenthetical information from title if present
        title=$(echo "$title_with_extra" | sed 's/ ([^)]*)//g' | sed 's/[[:space:]]*$//')
        
        echo "Processing: $filename"
        echo "Artist: $artist"
        echo "Title: $title"
        
        # Run the transcription
        python3 -m src.transcribe "$song" --artist "$artist" --track "$title"
        
        # If processing was successful, move the file to finished directory
        if [ $? -eq 0 ]; then
            # Move the original audio file to finished directory
            mv "$song" "finished/"
            
            # Copy the generated LRC file to finished directory
            output_dir="output/$artist - $title"
            
            # Check for LRC file in the main output directory
            if [ -f "$output_dir/lyrics.lrc" ]; then
                cp "$output_dir/lyrics.lrc" "finished/${filename%.flac}.lrc"
                echo "Copied LRC file to finished directory"
            # Check for LRC file in the subdirectory (some songs have different directory structure)
            elif [ -f "$output_dir/$artist - $title/lyrics.lrc" ]; then
                cp "$output_dir/$artist - $title/lyrics.lrc" "finished/${filename%.flac}.lrc"
                echo "Copied LRC file from subdirectory to finished directory"
            # Check for transcription.json and generate LRC if needed
            elif [ -f "$output_dir/transcription.json" ]; then
                echo "LRC file not found, but transcription.json exists. Generating LRC file..."
                python3 -m src.transcribe --lyrics-only --artist "$artist" --track "$title"
                
                if [ -f "$output_dir/lyrics.lrc" ]; then
                    cp "$output_dir/lyrics.lrc" "finished/${filename%.flac}.lrc"
                    echo "Generated and copied LRC file to finished directory"
                else
                    echo "Warning: Failed to generate LRC file"
                fi
            else
                echo "Warning: Neither LRC file nor transcription.json found in $output_dir"
            fi
            
            echo "Successfully processed and moved: $filename"
        else
            echo "Error processing: $filename"
        fi
    fi
done

# Deactivate virtual environment
deactivate
