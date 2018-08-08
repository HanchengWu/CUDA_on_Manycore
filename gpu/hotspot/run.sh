for i in 64 128 256 512 1024 2048; do
   for rep in `seq 1 5`; do	
	echo "./hotspot ${i} 2 4 ../../data/hotspot/temp_${i} ../../data/hotspot/power_${i} output.out"
	eval "./hotspot ${i} 2 4 ../../data/hotspot/temp_${i} ../../data/hotspot/power_${i} output.out"
   done
done

