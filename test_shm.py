import ctypes, sys
p = r"C:\Users\evanm\miniconda3\envs\hdr-dsp-sr\Lib\site-packages\torch\lib\shm.dll"
try:
    ctypes.CDLL(p)
    print("Loaded OK")
except Exception as e:
    print("Error:", e)
    sys.exit(1)
