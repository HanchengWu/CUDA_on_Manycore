for D in 1 2 3 4; do

	eval "./ctrl 1.0 1 1 32 100000 ${D} | tee -a result.txt"

done

for D in 1 2 3 4; do

	eval "./ctrl 1.0 1 12 16 100000 ${D} | tee -a result.txt"

done


for D in 1 2 3 4; do

	eval "./ctrl 1.0 2 24 16 100000 ${D} | tee -a result.txt"

done





