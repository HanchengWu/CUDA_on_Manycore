for i in 64 128 256 512 1024 2048; do
for aff in 2; do


   for rep in `seq 1 3`; do	
	echo "${seq} ./hotspot ${aff} 1.0 ${i} 2 4 ../../../data/hotspot/temp_${i} ../../../data/hotspot/power_${i} output.out"
	eval "./hotspot ${aff} 1.0 ${i} 2 4 ../../../data/hotspot/temp_${i} ../../../data/hotspot/power_${i} output.out"
   done


done
done

