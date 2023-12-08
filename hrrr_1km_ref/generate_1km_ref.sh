#!/bin/bash
rm ${PWD}/Northeast/*.png ; ${PWD}/Southeast/*.png ; ${PWD}/Northwest/*.png ; ${PWD}Southwest/*.png
python hrrr_ref_1km.py
node gen_html_new.js
git add -A
git commit -m 'HRRR run YYYYMMDDHH'
git push
