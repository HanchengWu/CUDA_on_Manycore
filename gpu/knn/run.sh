for i in 128 256 512 1024 2048 4096 8192; do
   for rep in `seq 1 5`; do	
	echo "${rep} ./knn list${i}k.txt -r 5 -lat 30 -lng 90"
	eval "./knn list${i}k.txt -r 5 -lat 30 -lng 90"
   done
done

