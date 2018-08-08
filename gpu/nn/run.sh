for i in 10000 20000 30000 40000 50000 60000 70000; do
   for rep in `seq 1 5`; do	
	echo "${rep} ./test ${i} 1 1 1"
	eval "./test ${i} 1 1 1"
   done
done

