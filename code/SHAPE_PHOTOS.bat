@echo off
setlocal
REM 26 Nov: PHOTO SESSION for the shape classifier - top camera only, the arms do NOT move.
REM For each shape: ONLY that shape on the table (all colours, small + large), spread out inside the black line,
REM Enter = photo, rearrange (turn them, lay down AND stand up), ~10 photos, then n = next shape. q = finish.
call "%~dp0..\config.bat"
REM   SHAPE_PHOTOS.bat cylinder   = start at cylinders (cube / cylinder / dodecahedron / pentagonal_prism)
python "%~dp0shape_photos.py" %1
pause
