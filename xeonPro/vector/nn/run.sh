for i in 10000 20000 30000 40000 50000 60000 70000; do
for aff in 1 2 3 4; do
   for rep in `seq 1 3`; do	
	echo "${rep} ./nn ${i} 1 1 1 ${aff} 1.0"
	eval "./nn ${i} 1 1 1 ${aff} 1.0"
   done
done
done

