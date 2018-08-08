for ite in `seq 1 3`; do

    for i in `seq 1 28` ;do
	for j in 32; do

		eval "./mem 32 1.0 1 ${i} ${j} 1 | tee -a time.txt"
	done
    done 

    eval "cat result_gpu.txt >> tmp"		
    eval "echo "" >> tmp"
    eval "rm result_gpu.txt"
done
