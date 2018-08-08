for i in 128 256 384 512 640 768 896 1024; do
for aff in 1 2 3 4; do
   for rep in `seq 1 5`; do	
	echo "${rep} ./gaussian -a ${aff} -f ../../../data/gaussian/matrix${i}.txt"
	eval "./gaussian -a ${aff} -f ../../../data/gaussian/matrix${i}.txt"
   done
done
done

