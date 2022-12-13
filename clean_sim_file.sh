#!/bin/bash
filename=/home/nicholas/PycharmProjects/curvePlot/Real/NACA\ 0018/xf-naca0018-il-200000.csv
sed 1,12d "$filename" > "$filename.temp1"
tr -s ' ' < "$filename.temp1" > "$filename.temp"
sed -i 's/^ *//' "$filename.temp"
tr ' ' ';' < "$filename.temp" > "$filename.csv"
rm "$filename.temp1" "$filename.temp"