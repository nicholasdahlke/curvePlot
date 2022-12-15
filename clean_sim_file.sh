#!/bin/bash
sed 1,12d "$1" > "$1.temp1"
tr -s ' ' < "$1.temp1" > "$1.temp"
sed -i 's/^ *//' "$1.temp"
tr ' ' ';' < "$1.temp" > "$1.csv"
rm "$1.temp1" "$1.temp" "$1"
exit 0