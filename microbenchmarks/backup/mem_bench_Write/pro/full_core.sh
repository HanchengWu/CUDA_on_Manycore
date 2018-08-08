for ite in `seq 1 5`; do

    eval "rm pro_result.txt"
    
    for X in 1 10 100 1000 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 200000 300000 400000 500000 600000 700000 800000 900000 1000000; do
	eval "./mem 2097152 1.0 1 64 16 ${X} "
    done

    eval "cat pro_result.txt >> mem_full_core.txt"		
    eval "echo "" >> mem_full_core.txt"
    eval "rm pro_result.txt"

done
