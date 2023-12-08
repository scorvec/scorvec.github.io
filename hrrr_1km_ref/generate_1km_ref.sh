#!/bin/bash
rm ${PWD}/Northeast/*.png ; ${PWD}/Southeast/*.png ; ${PWD}/Northwest/*.png ; ${PWD}/Southwest/*.png
python hrrr_ref_1km.py
node gen_html_new.js
sed 's|/home/shawn/1km_ref/|/|g' anim.html > index.html
git add -A
git commit -m 'HRRR run YYYYMMDDHH'
git push
