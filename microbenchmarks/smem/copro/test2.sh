
for ite in `seq 1 5`; do

    eval "rm result.txt"

    for X in 1 2 3 4; do
	eval "./mem 2097152 1.0 ${X} 1 64 512 "
    done

    eval "cat result.txt >> smem2.txt"		
    eval "echo "" >> smem2.txt"
    eval "rm result.txt"

done
