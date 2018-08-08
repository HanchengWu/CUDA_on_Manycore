
make clean; make -j

for T in `seq 1 120` ;do
  for i in `seq 1 20` ;do
		echo "${i} ./mem 32 1.0 1 ${T} 32 1"
		eval "./mem 32 1.0 1 ${T} 32 1"
	done
done
