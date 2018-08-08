
PACKET_SIZE="500"


#----------------------------------------------Fermi-------------------------------------------------

echo "Fermi_2048patterns_kernel"
OUTFILE="Fermi_traversal.out"
STRUCTURAL_CONFIG="--automata_file ../local/MyWorkspace/git_repos/Fixed_Topology_Compiler/data/mem_layout/Fermi_2048patterns_mem.txt"
TRACE_CONFIG="--trace_num 1 --tracefile_names ../local/MyWorkspace/git_repos/Fixed_Topology_Compiler/data/trace_file/fermi_rp_input_10MB.input --pkt_num 1"
OCCUPANCY_STREAM_NUM="5" #in reality its the number of streams
OCCUPANCY_STREAM_NUM_CONFIG='--stream_count '$OCCUPANCY_STREAM_NUM' '


echo "./regex_gpu $STRUCTURAL_CONFIG $TRACE_CONFIG --pkt_size $PACKET_SIZE --device 0 $OCCUPANCY_STREAM_NUM_CONFIG > $OUTFILE 2>&1"
./regex_gpu --general $STRUCTURAL_CONFIG $TRACE_CONFIG --pkt_size $PACKET_SIZE --device 0 $OCCUPANCY_STREAM_NUM_CONFIG > $OUTFILE 2>&1
