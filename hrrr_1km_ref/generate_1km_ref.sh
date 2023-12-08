#!/bin/bash
rm ${PWD}/Northeast/*.png ;rm ${PWD}/Southeast/*.png ;rm ${PWD}/Northwest/*.png ;rm ${PWD}/Southwest/*.png
python hrrr_ref_1km.py
node gen_html_new.js
sed 's|/home/shawn/public_html/hrrr_1km_ref/||g' anim.html > index.html
git add -A
git commit -m 'HRRR run YYYYMMDDHH'
git push
