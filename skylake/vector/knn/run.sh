for i in 128 256 512 1024 2048 4096 8192; do
  for aff in 1 2; do
        for rep in `seq 1 10`; do	
		echo "${rep} ./knn list${i}k.txt -r 5 -lat 30 -lng 90 -a ${aff}"
		eval "./knn list${i}k.txt -r 5 -lat 30 -lng 90 -a ${aff}"
   	done
  done
done

