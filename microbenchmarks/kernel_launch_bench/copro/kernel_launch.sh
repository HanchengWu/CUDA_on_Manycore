
for i in `seq 1 28` ;do
	for j in 32; do

		eval "./mem 32 1.0 1 ${i} ${j} 1 | tee -a kernel_launch_copro.txt"

	done
done
