for ite in `seq 1 5`; do

    eval "rm gpu_result.txt"

#    for X in 1 10 100 1000 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 ; do
    for X in 1 2; do

	eval "./sync 100000 1.0 1 1 32 ${X}"

    done

    eval "cat gpu_result.txt >> single_warp.txt"		
    eval "echo "" >> single_warp.txt"
    eval "rm gpu_result.txt"

done
