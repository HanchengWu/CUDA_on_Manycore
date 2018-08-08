
for ite in `seq 1 5`; do

    eval "rm result.txt"

    for X in 1 128 256 384 512 640 768 896 1024; do
	eval "./mem 2097152 1.0 1 1 64 ${X} "
    done

    eval "cat result.txt >> smem.txt"		
    eval "echo "" >> smem.txt"
    eval "rm result.txt"

done
