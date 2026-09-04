"""Measure caller->agent turn gaps from a Twilio dual-channel recording.

Ground truth: this is the audio as it existed on the phone line, so it
includes every leg (tunnel, our process, OpenAI) that a felt_s measured
inside our own process cannot see.
"""
import sys, glob, os
import soundfile as sf, numpy as np

FRAME = 0.02
ON, OFF = 0.020, 0.010      # hysteresis on frame RMS
MIN_SPEECH = 0.20           # ignore blips shorter than this
MIN_GAP    = 0.15           # bridge pauses shorter than this inside one turn

def segments(x, sr):
    W = int(FRAME*sr); n = len(x)//W
    r = np.sqrt((x[:n*W].reshape(n,W).astype(np.float64)**2).mean(axis=1))
    on = False; segs=[]; start=0
    for i,v in enumerate(r):
        if not on and v > ON: on=True; start=i
        elif on and v < OFF: on=False; segs.append((start*FRAME, i*FRAME))
    if on: segs.append((start*FRAME, n*FRAME))
    # bridge short gaps, then drop short segments
    merged=[]
    for s,e in segs:
        if merged and s-merged[-1][1] < MIN_GAP: merged[-1]=(merged[-1][0], e)
        else: merged.append((s,e))
    return [(s,e) for s,e in merged if e-s >= MIN_SPEECH]

def analyse(path, verbose=True):
    d,sr = sf.read(path, always_2d=True)
    if d.shape[1] < 2: return None
    caller = segments(d[:,0], sr)
    agent  = segments(d[:,1], sr)
    gaps=[]
    for cs,ce in caller:
        nxt = next((a for a in agent if a[0] >= ce - 0.02), None)
        if nxt is None: continue
        # ignore if the caller starts again before the agent does (caller
        # kept talking; that gap is not a wait)
        later_caller = next((c for c in caller if c[0] > ce + 0.02), None)
        if later_caller and later_caller[0] < nxt[0] - 0.05: continue
        gaps.append((ce, nxt[0]-ce))
    if verbose:
        print(f"\n{os.path.basename(path)}  dur {d.shape[0]/sr:.1f}s  "
              f"caller turns {len(caller)}  agent turns {len(agent)}")
        for ce,g in gaps:
            bar = '#'*int(g*10)
            print(f"   caller ends {ce:7.2f}s -> agent starts after {g:5.2f}s  {bar}")
    return [g for _,g in gaps]

if __name__ == "__main__":
    pats = sys.argv[1:] or ["data/3 cases voice/twilio-*.mp3"]
    allg=[]
    for pat in pats:
        for p in sorted(glob.glob(pat)):
            g = analyse(p, verbose=(len(pats)==1 and '*' not in pat))
            if g: allg += g
    if allg:
        a=np.array(allg); a.sort()
        print(f"\nALL GAPS n={len(a)}  median {np.median(a):.2f}s  mean {a.mean():.2f}s  "
              f"p90 {np.percentile(a,90):.2f}s  max {a.max():.2f}s")
