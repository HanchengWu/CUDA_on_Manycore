
for i in `seq 1 28` ;do
	for j in 32 64 96 128 ; do

		eval "./mem 32 1.0 2 ${i} ${j} 1 | tee -a kernel_launch_overhead.txt"

	done
done
