eval "make clean; make sync_only;"
for ite in `seq 1 5`; do

    eval "rm result.txt"

    for X in 16 32 48 64 80 96 112 128  ; do

	eval "./sync 100000 1.0 1 1 ${X} 1"

    done

    eval "cat result.txt >> increase_block.txt"		
    eval "echo "" >> increase_block.txt"
    eval "rm result.txt"

done
