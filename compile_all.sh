#!/bin/bash

for (( i = 0; i < 7; i++ )); do
	d="0$((${i}+1))"

	echo "Compiling slide ${d}"
	cd ${d} 

	latexmk -lualatex -interaction=nonstopmode $f 2>/dev/null 1>/dev/null && \
		echo "Succes" || echo "Failure"
	cd ..

done
