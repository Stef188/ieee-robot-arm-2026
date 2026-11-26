BIN LAYOUTS - colour sorter (26 Nov 2026)

ONE model covers every order of the 4 bins. No separate model per layout is needed.
The arm drop poses belong to the 4 bin SPOTS (where the bins stood in our demos), not to the colours:
whatever colour stands on a spot, the arm next to that spot drops it there, with that spot's pose, drop height and
learned sag correction (model.ArmMap.bin_slot, sort.py plan()).

Our 4 spots (top camera, left -> right):  red spot (224,265) left arm | green spot (305,250) left arm |
                                           blue spot (408,256) right arm | yellow spot (502,280) right arm
-> ALL 24 orders of the 4 colours on these spots work: see ALL_24_LAYOUTS.txt (made by check_layout.py all).
   Example: the organisers' picture (red and yellow swapped vs our lab): left arm drops yellow + green, right arm red + blue.

SETUP DAY
1. Put the bins where the organisers want them.
2. Run  classical\CHECK_BINS.bat  (camera only, arms don't move). It prints, per colour, which arm drops it - or
   "NOT ON A KNOWN SPOT" - and saves live_check.png with the bins and our spots marked.
3. If a bin is NOT on a known spot (more than ~4 cm from all 4 spots):
     - nudge it onto the nearest spot if the organisers allow it, or
     - hand-teach that bin: TEACH_BIN.bat <left|right> <colour>  (the arm goes limp, hold the gripper ~5 cm above the
       middle of the bin, Enter). A taught bin is always used exactly.
   The general fix for bins anywhere is a learned drop-pose model (not built yet).

Files: check_layout.py (the checks), ALL_24_LAYOUTS.txt (proof all orders work), image_check.png (example output).
