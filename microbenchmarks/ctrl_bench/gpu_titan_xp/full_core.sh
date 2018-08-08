
for ite in `seq 1 5`; do

    eval "rm result.txt"
    for D in 1 2 3 4; do
	eval "./ctrl 1.0 1 30 128 100000 ${D} "
    done

    eval "cat result.txt >> full_core.txt"		
    eval "echo "" >> full_core.txt"
    eval "rm result.txt"

done
