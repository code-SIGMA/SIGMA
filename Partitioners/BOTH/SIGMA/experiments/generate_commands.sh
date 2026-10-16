#!/bin/bash

# Directory containing graph files
graph_test_dir="/home/shaiperetz/HiWiAlgEng/gnn_test_graphs/"

partitions=( 2 4 8 16 32 64 128 )

# Output file to store commands
output_file="par_commands.txt"

# Clear the output file if it already exists
echo -n "" > "$output_file"

# List of graph files

graph_test=(
    "amazon_computers.metis"
    "flickr.metis"
    "arxiv.metis"
    # "ogbn-papers100M.graph"
    #"ogbn-products.graph"
    #"reddit.graph"
    "twitch.metis"
    #"yelp.graph"
)

# Loop through each mode and graph

# ==== LOOP OVER ALL GRAPHS AND CONFIGURATIONS ====
for graph in "${graph_test[@]}"; do
    for k in "${partitions[@]}"; do
        input_graph="${graph_test_dir}${graph}"
        base_output_path="/home/shaiperetz/HiWiAlgEng/mult_constr_mult_obj/MultiConstraintPartitioning/experiments/hyp_test"

        # --- Base configuration ---
        # output_path="${base_output_path}/mod_fennel_light_plus/"
        # mkdir -p "$output_path"
        # command="/home/speretz/HiWi/v2/MultiConstraintPartitioning/build/sigma $input_graph --k=$k --write_results --one_pass_algorithm=mod_fennel --output_path=$output_path --suppress_file_output --prepartitioner=clustre --mode=light_plus --partitioning_type=node"
        # echo "$command" >> "$output_file"

        # output_path="${base_output_path}/fennel_mult_obj_light_plus/"
        # mkdir -p "$output_path"
        # command="/home/speretz/HiWi/v2/MultiConstraintPartitioning/build/sigma $input_graph --k=$k --write_results --one_pass_algorithm=fennel_mult_obj --output_path=$output_path --suppress_file_output --prepartitioner=clustre --mode=light_plus --partitioning_type=node"
        # echo "$command" >> "$output_file"

        # output_path="${base_output_path}/hdrf_light_plus/"
        # mkdir -p "$output_path"
        # command="/home/speretz/HiWi/v2/MultiConstraintPartitioning/build/sigma $input_graph --k=$k --write_results --one_pass_algorithm=hdrf --output_path=$output_path --suppress_file_output --prepartitioner=clustre --mode=light_plus --partitioning_type=edge"
        # echo "$command" >> "$output_file"

        # output_path="${base_output_path}/mod_fennel_strong/"
        # mkdir -p "$output_path"
        # command="/home/speretz/HiWi/v2/MultiConstraintPartitioning/build/sigma $input_graph --k=$k --write_results --one_pass_algorithm=mod_fennel --output_path=$output_path --suppress_file_output --prepartitioner=clustre --mode=strong --partitioning_type=node"
        # echo "$command" >> "$output_file"

        # output_path="${base_output_path}/fennel_mult_obj_strong/"
        # mkdir -p "$output_path"
        # command="/home/speretz/HiWi/v2/MultiConstraintPartitioning/build/sigma $input_graph --k=$k --write_results --one_pass_algorithm=fennel_mult_obj --output_path=$output_path --suppress_file_output --prepartitioner=clustre --mode=strong --partitioning_type=node"
        # echo "$command" >> "$output_file"

        # output_path="${base_output_path}/hdrf_strong/"
        # mkdir -p "$output_path"
        # command="/home/speretz/HiWi/v2/MultiConstraintPartitioning/build/sigma $input_graph --k=$k --write_results --one_pass_algorithm=hdrf --output_path=$output_path --suppress_file_output --prepartitioner=clustre --mode=strong --partitioning_type=edge"
        # echo "$command" >> "$output_file"


        # output_path="${base_output_path}/mod_fennel_light_plus/"
        # mkdir -p "$output_path"
        # command="/home/speretz/HiWi/MCP/MultiConstraintPartitioning/build/sigma $input_graph --k=$k --write_results --one_pass_algorithm=mod_fennel --output_path=$output_path --suppress_file_output --prepartitioner=clustre --mode=light_plus --partitioning_type=node"
        # echo "$command" >> "$output_file"

        output_path="${base_output_path}/fennel_mult_obj_${k}_25/"
        mkdir -p "$output_path"
        command="/home/shaiperetz/HiWiAlgEng/mult_constr_mult_obj/MultiConstraintPartitioning/build/sigma $input_graph --k=$k --write_results --one_pass_algorithm=fennel_mult_obj --output_path=$output_path --suppress_file_output --prepartitioner=clustre --mode=light_plus --partitioning_type=node"
        echo "$command" >> "$output_file"

        output_path="${base_output_path}/fennel_mult_obj_${k}_15/"
        mkdir -p "$output_path"
        command="/home/shaiperetz/HiWiAlgEng/mult_constr_mult_obj/MultiConstraintPartitioning/build/sigma $input_graph --k=$k --write_results --one_pass_algorithm=fennel_mult_obj --output_path=$output_path --suppress_file_output --prepartitioner=clustre --mode=light_plus --partitioning_type=node --fennel_gamma=1.5"
        echo "$command" >> "$output_file"

        # output_path="${base_output_path}/mod_fennel_strong/"
        # mkdir -p "$output_path"
        # command="/home/speretz/HiWi/MCP/MultiConstraintPartitioning/build/sigma $input_graph --k=$k --write_results --one_pass_algorithm=mod_fennel --output_path=$output_path --suppress_file_output --prepartitioner=clustre --mode=strong --partitioning_type=node"
        # echo "$command" >> "$output_file"

        # output_path="${base_output_path}/fennel_mult_obj_strong/"
        # mkdir -p "$output_path"
        # command="/home/speretz/HiWi/MCP/MultiConstraintPartitioning/build/sigma $input_graph --k=$k --write_results --one_pass_algorithm=fennel_mult_obj --output_path=$output_path --suppress_file_output --prepartitioner=clustre --mode=strong --partitioning_type=node"
        # echo "$command" >> "$output_file"

        # output_path="${base_output_path}/hdrf_strong/"
        # mkdir -p "$output_path"
        # command="/home/speretz/HiWi/MCP/MultiConstraintPartitioning/build/sigma $input_graph --k=$k --write_results --one_pass_algorithm=hdrf --output_path=$output_path --suppress_file_output --prepartitioner=clustre --mode=strong --partitioning_type=edge"
        # echo "$command" >> "$output_file"


        # --- With prepartitioner and artificial weights ---
        # output_path="${base_output_path}/new/"
        # mkdir -p "$output_path"
        # command="../build/clustre $input_graph --k=$k --write_results --one_pass_algorithm=range --output_path=$output_path --suppress_file_output --prepartitioner=clustre --mode=light_plus --with_aw --partitioning_type=node"
        # echo "$command" >> "$output_file"
    done
done

# Inform user of completion
echo "Commands written to $output_file"