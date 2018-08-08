for X in 1 10 100 1000 10000 20000 30000 40000 50000; do

	eval "./sync 100000 1.0 1 1 32 ${X} | tee -a result_single_warp.txt"

done
