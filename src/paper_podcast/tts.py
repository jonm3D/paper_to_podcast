"""Text-to-speech using ElevenLabs API."""

import os
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs import ElevenLabs


def get_client() -> ElevenLabs:
    """Get an ElevenLabs client using the API key from environment or .env file."""
    load_dotenv()

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError(
            "ELEVENLABS_API_KEY not set. Either:\n"
            "  1. Set environment variable: export ELEVENLABS_API_KEY=...\n"
            "  2. Add to .env file: ELEVENLABS_API_KEY=..."
        )
    return ElevenLabs(api_key=api_key)


def list_voices() -> list[dict]:
    """List available ElevenLabs voices."""
    client = get_client()
    response = client.voices.get_all()
    return [{"name": v.name, "voice_id": v.voice_id} for v in response.voices]


# Default voice IDs (these are standard ElevenLabs voices)
VOICES = {
    "rachel": "21m00Tcm4TlvDq8ikWAM",  # Calm, professional female
    "domi": "AZnzlk1XvdvUeBnXmlld",     # Young female
    "bella": "EXAVITQu4vr4xnSDxMaL",    # Soft female
    "antoni": "ErXwobaYiN019PkySvjV",   # Warm male
    "josh": "TxGEqnHWrfWFTfGW9XjX",     # Deep male
    "adam": "pNInz6obpgDQGcFmaJgB",     # Deep male narrator
    "sam": "yoZ06aMxZJJ28mfd3POQ",      # Young male
}


def generate_audio(
    text: str,
    output_path: Path,
    voice: str = "rachel",
    model: str = "eleven_multilingual_v2",
) -> Path:
    """Generate audio from text using ElevenLabs.

    Args:
        text: The text to convert to speech.
        output_path: Path to save the audio file.
        voice: Voice name (rachel, adam, josh, etc.) or voice ID.
        model: ElevenLabs model to use.

    Returns:
        Path to the saved audio file.
    """
    client = get_client()

    # Resolve voice name to ID
    voice_id = VOICES.get(voice.lower(), voice)

    # Generate audio
    audio_generator = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id=model,
    )

    # Collect audio bytes from generator
    audio_bytes = b"".join(audio_generator)

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(audio_bytes)

    return output_path


def generate_podcast_script(paper_markdown: str, title: str) -> str:
    """Generate a two-host podcast discussion script for PhD qual prep.

    Creates a rigorous academic discussion between two hosts analyzing
    the paper's methodology, findings, significance, and limitations.

    Args:
        paper_markdown: The cleaned paper markdown.
        title: Paper title.

    Returns:
        A dialogue script with HOST_A and HOST_B labels.
    """
    import anthropic
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""Create a 5-10 minute podcast discussion between two researchers reviewing this academic paper for PhD qualifying exam preparation.

FORMAT:
- Two hosts: HOST_A and HOST_B
- Each line starts with "HOST_A:" or "HOST_B:"
- Natural academic conversation, not a formal presentation
- Hosts should build on each other's points, ask clarifying questions, and occasionally disagree

CONTENT STRUCTURE:
1. Opening (30 sec): Brief intro to the paper - what's the core research question?
2. Context & Motivation (1-2 min): Why does this problem matter? What gap does it fill?
3. Methodology Deep Dive (2-3 min): How did they approach this? What are the key technical choices? Discuss strengths and limitations of the methods.
4. Key Results (2-3 min): What did they find? Reference specific figures/tables. Discuss statistical significance and effect sizes where relevant.
5. Critical Analysis (1-2 min): Limitations, assumptions, potential confounds. What would you do differently?
6. Broader Impact (1 min): How does this fit into the field? What are the implications? Future directions?

TONE:
- PhD-level rigor - don't oversimplify
- Use proper technical terminology
- Be critical but fair
- Discuss methodology choices substantively
- Reference specific results, numbers, and figures
- Natural conversation with occasional "hmm", "right", "exactly" but not excessive

Paper title: {title}

Paper content:
{paper_markdown}

Generate the discussion:"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text


def parse_dialogue(script: str) -> list[tuple[str, str]]:
    """Parse a dialogue script into speaker/text pairs.

    Args:
        script: Script with HOST_A: and HOST_B: labels.

    Returns:
        List of (speaker, text) tuples.
    """
    import re

    lines = []
    current_speaker = None
    current_text = []

    for line in script.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Check for speaker label
        match = re.match(r'^(HOST_[AB]):\s*(.*)$', line)
        if match:
            # Save previous speaker's text
            if current_speaker and current_text:
                lines.append((current_speaker, ' '.join(current_text)))
            current_speaker = match.group(1)
            current_text = [match.group(2)] if match.group(2) else []
        elif current_speaker:
            # Continuation of current speaker
            current_text.append(line)

    # Don't forget the last speaker
    if current_speaker and current_text:
        lines.append((current_speaker, ' '.join(current_text)))

    return lines


def generate_dialogue_audio(
    script: str,
    output_path: Path,
    voice_a: str = "josh",      # Male voice for Host A
    voice_b: str = "rachel",    # Female voice for Host B
    model: str = "eleven_multilingual_v2",
) -> Path:
    """Generate audio from a two-host dialogue script.

    Args:
        script: Dialogue script with HOST_A/HOST_B labels.
        output_path: Path to save the combined audio file.
        voice_a: Voice for Host A.
        voice_b: Voice for Host B.
        model: ElevenLabs model to use.

    Returns:
        Path to the saved audio file.
    """
    client = get_client()

    # Parse the dialogue
    dialogue = parse_dialogue(script)
    if not dialogue:
        raise ValueError("Could not parse dialogue from script")

    print(f"    Parsed {len(dialogue)} dialogue segments")

    # Generate audio for each segment
    audio_segments = []
    for i, (speaker, text) in enumerate(dialogue):
        voice = voice_a if speaker == "HOST_A" else voice_b
        voice_id = VOICES.get(voice.lower(), voice)

        print(f"    Segment {i+1}/{len(dialogue)} ({speaker})...")

        audio_generator = client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id=model,
        )
        audio_bytes = b"".join(audio_generator)
        audio_segments.append(audio_bytes)

    # Combine all segments
    combined_audio = b"".join(audio_segments)

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(combined_audio)

    return output_path


# Test when run directly
if __name__ == "__main__":
    print("Available voices:")
    try:
        voices = list_voices()
        for v in voices[:10]:
            print(f"  - {v['name']} ({v['voice_id']})")
    except Exception as e:
        print(f"  Error: {e}")
        print("  Set ELEVENLABS_API_KEY to see available voices")
