#for i in 128 256 384 512 640 768 896 1024; do
for i in 128; do
   for rep in `seq 1 5`; do	
	echo "${rep} ./gaussian -f ../../data/gaussian/matrix${i}.txt"
	eval "./gaussian -f ../../data/gaussian/matrix${i}.txt"
   done
done

