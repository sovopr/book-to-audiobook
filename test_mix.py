import os, subprocess, glob
files = sorted(glob.glob("output/tmp/*/*.mp3"))[:10]
with open("test.txt", "w") as f:
    for file in files:
        f.write(f"file '{os.path.abspath(file)}'\n")
print(f"Testing concat on {len(files)} files")
try:
    res = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "test.txt", "-c", "copy", "test_out.mp3"], capture_output=True, text=True, check=True)
    print("Success!")
except subprocess.CalledProcessError as e:
    print("FAILED with code", e.returncode)
    print(e.stderr)
