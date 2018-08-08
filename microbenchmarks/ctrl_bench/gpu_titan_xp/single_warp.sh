
for ite in `seq 1 5`; do

    eval "rm result.txt"
    for D in 1 2 3 4; do
	eval "./ctrl 1.0 1 1 32 100000 ${D} "
    done

    eval "cat result.txt >> single_warp.txt"		
    eval "echo "" >> single_warp.txt"
    eval "rm result.txt"

done
