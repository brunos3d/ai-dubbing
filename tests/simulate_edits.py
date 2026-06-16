import json
import shutil
import sys
from pathlib import Path
import subprocess

def run_generate(wid, args=[]):
    cmd = ["./dub.sh", "generate", wid] + args
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result

def get_ran_stages(output):
    # Heuristic: look for stage banners in output.  The stage banner is
    # logged on the rich console (stderr) and the file handler (which
    # is not captured by subprocess).  The captured stderr from
    # dub.sh will include the stage banner (formatted as
    # ``[X/Y] [X/Y] <Stage Name>`` -- the first X/Y is the rich
    # handler's %(stage)s attribute, the second is the message itself).
    # Earlier versions of this script read only ``res.stdout``; the
    # Rich console writes to stderr, so banners were never seen and
    # the test always reported "no stages ran".  We now merge stderr
    # and stdout into a single buffer before scanning.
    #
    # The rich console also wraps the digits in ANSI colour escapes
    # (``[1;36m8[0m/[1;36m11[0m``), so the literal regex
    # ``\[\\d+/11\]`` does not match.  Strip ANSI escapes first.
    import re
    ansi_re = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
    text = ansi_re.sub("", output)
    stages = []
    matches = re.findall(r"\[\d+/11\]\s*\[?\d*/11\]?\s*(.*)", text)
    for m in matches:
        s = m.strip().lower()
        if "extraction" in s: stages.append("extract")
        elif "separation" in s: stages.append("separate")
        elif "diarization" in s: stages.append("diarize")
        elif "sampling" in s or "sample extraction" in s: stages.append("samples")
        elif "transcription" in s: stages.append("transcribe")
        elif "translation" in s: stages.append("translate")
        elif "generation" in s or "omnivorce" in s or "tts" in s: stages.append("generate")
        elif "alignment" in s: stages.append("align")
        elif "reconstruction" in s: stages.append("reconstruct")
        elif "mix" in s: stages.append("mix")
        elif "video" in s: stages.append("video")
    return stages

def test_a(wid, root):
    print("\n--- TEST A: Edit translated_transcript.json ---")
    p = root / "translation" / "translated_transcript.json"
    data = json.loads(p.read_text())
    data[0]["text"] += " (edited)"
    p.write_text(json.dumps(data))

    res = run_generate(wid)
    stages = get_ran_stages((res.stdout or "") + (res.stderr or ""))
    print(f"Stages ran: {stages}")
    # Expected: generate, align, reconstruct, mix, video
    assert "translate" not in stages
    assert "generate" in stages

def test_b(wid, root):
    print("\n--- TEST B: Edit glossary.json ---")
    p = root / "translation" / "glossary.json"
    # Create if missing (it might be missing if not passed to prepare)
    if not p.exists():
        p.write_text(json.dumps({"entities": []}))
    else:
        data = json.loads(p.read_text())
        data["entities"].append({"original": "Dummy", "replacement": "Dummy", "action": "preserve"})
        p.write_text(json.dumps(data))

    res = run_generate(wid)
    stages = get_ran_stages((res.stdout or "") + (res.stderr or ""))
    print(f"Stages ran: {stages}")
    # Expected: translate, generate, align, reconstruct, mix, video
    assert "translate" in stages

def test_c(wid, root):
    print("\n--- TEST C: Replace a speaker primary sample ---")
    # Find a speaker wav file (samples stage writes the canonical
    # reference under speaker_profiles/.../reference.wav and then
    # copies it to the flat alias speakers/{spk}.wav; the DAG
    # tracks the flat alias as the editable input to generate).
    wav_files = sorted((root / "speakers").glob("*.wav"))
    assert wav_files, "no speaker wav files in workspace"
    p = wav_files[0]
    # Append a null byte so the on-disk hash differs from the
    # manifest's recorded hash for this file.
    p.write_bytes(p.read_bytes() + b"\0")

    res = run_generate(wid)
    stages = get_ran_stages((res.stdout or "") + (res.stderr or ""))
    print(f"Stages ran: {stages}")
    # Expected: generate, align, reconstruct, mix, video
    assert "translate" not in stages
    assert "generate" in stages

def test_d(wid, root):
    print("\n--- TEST D: Edit transcription/transcript.json ---")
    p = root / "transcription" / "transcript.json"
    data = json.loads(p.read_text())
    data[0]["text"] += " (edited)"
    p.write_text(json.dumps(data))

    res = run_generate(wid)
    stages = get_ran_stages((res.stdout or "") + (res.stderr or ""))
    print(f"Stages ran: {stages}")
    # Expected: translate, generate, align, reconstruct, mix, video
    assert "translate" in stages

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python simulate_edits.py <wid> <root>")
        sys.exit(1)
    wid = sys.argv[1]
    root = Path(sys.argv[2])
    
    # Run tests sequentially
    # We should probably 'clean' or 'reset' between tests if they depend on each other,
    # but here each edit invalidates downstream, so it's fine.
    
    # Pre-run to ensure clean state
    print("Ensuring clean state...")
    run_generate(wid)
    
    test_a(wid, root)
    test_b(wid, root)
    test_c(wid, root)
    test_d(wid, root)
