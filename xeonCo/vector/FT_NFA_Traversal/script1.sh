PACKET_SIZE="15000"
TRACE_PATH="../local/MyWorkspace/git_repos/Fixed_Topology_Compiler/data/trace_file/synthetic/multi_process/"

#----------------------------------------------Fermi-------------------------------------------------

echo "Fermi_2048patterns_kernel"
OUTFILE="Fermi_traversal.out"
STRUCTURAL_CONFIG="--automata_file ../local/MyWorkspace/git_repos/Fixed_Topology_Compiler/data/mem_layout/Fermi_2048patterns_mem.txt"
TRACE_CONFIG='--pkt_num 1 --trace_num 8 --tracefile_names '$TRACE_PATH'Fermi_2048patterns_depth_s0_p0.50.trace '$TRACE_PATH'Fermi_2048patterns_depth_s1_p0.50.trace '$TRACE_PATH'Fermi_2048patterns_depth_s2_p0.50.trace '$TRACE_PATH'Fermi_2048patterns_depth_s3_p0.50.trace '$TRACE_PATH'Fermi_2048patterns_depth_s4_p0.50.trace '$TRACE_PATH'Fermi_2048patterns_depth_s5_p0.50.trace '$TRACE_PATH'Fermi_2048patterns_depth_s6_p0.50.trace '$TRACE_PATH'Fermi_2048patterns_depth_s7_p0.50.trace'
OCCUPANCY_STREAM_NUM="8" #in reality its the number of streams
OCCUPANCY_STREAM_NUM_CONFIG='--stream_count '$OCCUPANCY_STREAM_NUM' '


echo "./regex_xeon $STRUCTURAL_CONFIG $TRACE_CONFIG --pkt_size $PACKET_SIZE --device 0 $OCCUPANCY_STREAM_NUM_CONFIG > $OUTFILE 2>&1"
./regex_xeon --general $STRUCTURAL_CONFIG $TRACE_CONFIG --pkt_size $PACKET_SIZE --device 0 $OCCUPANCY_STREAM_NUM_CONFIG > $OUTFILE 2>&1
