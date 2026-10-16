/******************************************************************************
 * linear_obj_edge.cpp
 * *
 * Source of KaHIP -- Karlsruhe High Quality Partitioning.
 *****************************************************************************/

#include "linear_obj_edge.h"
#include "partition/onepass_partitioning/floating_block.h"
#include "partition/onepass_partitioning/vertex_partitioning.h"

onepass_linear_obj_edge::onepass_linear_obj_edge(PartitionID k0, PartitionID kf,
                               PartitionID max_blocks, NodeID n_threads,
                               bool hashing /*=false*/, float gamma /*=1.5*/)
    : vertex_partitioning(k0, kf, max_blocks, n_threads, hashing) {
    this->gamma = gamma;
}

onepass_linear_obj_edge::~onepass_linear_obj_edge() {}

double onepass_linear_obj_edge::compute_score(floating_block &block, int my_thread, HeiClus::PartitionConfig &config, PartitionID cluster, int restreaming, LongNodeID curr_node_id, std::vector<floating_block> & blocks, std::pair<NodeID, EdgeWeight> const& neighbour_node) {
    
    // to make the linear objective function work I require to get the com2part and the volume of the communities

	// e.first = curr_node_id; e.second = neighbour_node.first
    
    EdgeWeight degree_curr_node = (*config.node_degrees)[curr_node_id]; // degree_u
    EdgeWeight degree_neighbour_node = (*config.node_degrees)[neighbour_node.first]; //degree_v
    
    // NEED TO CHANGE: get vertex clusters! if target partitions based on vertex clusters are already full
    PartitionID partition_curr_node = (*config.stream_nodes_assign)[curr_node_id]; 
    PartitionID partition_neighbour_node = (*config.stream_nodes_assign)[neighbour_node.first]; 
    // This is how it is done in 2PS:
    // uint32_t com_part_u = com2part[communities[e.first]];
    // uint32_t com_part_v = com2part[communities[e.second]];
	
    std::vector<NodeWeight> edge_load_curr_node;
	edge_load_curr_node.resize(config.number_of_constraints); 
    
    edge_load_curr_node[0] = 0;
    edge_load_curr_node[1] = 0;

    // Adjust the replication node weight associated with assigning edge to node_partition 
    if(!(*config.stream_edges_assign)[curr_node_id][partition_curr_node]) {
        edge_load_curr_node[0]++;
    }
    if(!(*config.stream_edges_assign)[neighbour_node.first][partition_curr_node]) {
        edge_load_curr_node[0]++;
    }
    edge_load_curr_node[1] = neighbour_node.second; // the edge weight


    std::vector<NodeWeight> edge_load_neighbour_node;
	edge_load_neighbour_node.resize(config.number_of_constraints); 

    edge_load_neighbour_node[0] = 0;
    edge_load_neighbour_node[1] = 0;

    // Adjust the weights associated with assigning edge to node_partition 
    if(!(*config.stream_edges_assign)[curr_node_id][partition_neighbour_node]) {
        edge_load_neighbour_node[0]++;
    }
    if(!(*config.stream_edges_assign)[neighbour_node.first][partition_neighbour_node]) {
        edge_load_neighbour_node[0]++;
    }
    edge_load_neighbour_node[1] = neighbour_node.second;


    uint64_t sum, sum_of_volumes;
    uint64_t external_degrees_u, external_degrees_v;
    double max_score = 0;
    PartitionID max_p = -1;
    double bal, gv, gu, gv_c, gu_c;

    std::vector<float> part_load(config.number_of_constraints);

    // if target partitions based on vertex clusters are already full
    if (blocks[partition_curr_node].fully_loaded_edge_multi_constraints(edge_load_curr_node) || blocks[partition_neighbour_node].fully_loaded_edge_multi_constraints(edge_load_neighbour_node)){
    	if (degree_curr_node > degree_neighbour_node){
    		max_p = curr_node_id % config.k;
    	}
    	else {
    		max_p = neighbour_node.first % config.k;
    	}

        std::vector<NodeWeight> edge_load_max_part;
        edge_load_max_part.resize(config.number_of_constraints); 

        edge_load_max_part[0] = 0;
        edge_load_max_part[1] = 0;

        // Adjust the weights associated with assigning edge to node_partition 
        if(!(*config.stream_edges_assign)[curr_node_id][max_p]) {
            edge_load_max_part[0]++;
        }
        if(!(*config.stream_edges_assign)[neighbour_node.first][max_p]) {
            edge_load_max_part[0]++;
        }
        edge_load_max_part[1] = neighbour_node.second;

        if (blocks[max_p].fully_loaded_edge_multi_constraints(edge_load_max_part)){
    		// assign to min loaded partition
            EdgeWeight min_load = std::numeric_limits<EdgeWeight>::infinity();
    		PartitionID min_p = 0;

    		for (PartitionID i = 0; i < config.k; i++){
                // consider only edge weights
                part_load[1] = static_cast<float>(blocks[i].partition_weights[1]) / static_cast<float>(blocks[i].partition_constraints[1]);
                
                // float max_overload;
                // if(part_load[0] > part_load[1]) {
                //     max_overload = part_load[0];
                // } else {
                //     max_overload = part_load[1];
                // }

                if (part_load[1] < min_load){
    				min_load = part_load[1];
    				min_p = i;
    			}
    		}
    		max_p = min_p;
    	}

    	return max_p;
    } else{
    	std::vector<PartitionID> ps{partition_curr_node, partition_neighbour_node};
        
        std::vector<NodeWeight> edge_load_p;
	    edge_load_p.resize(config.number_of_constraints); 

    	for (PartitionID p : ps) {
            edge_load_p[0] = 0;
            edge_load_p[1] = 0;

            // Adjust the weights associated with assigning edge to node_partition 
            if(!(*config.stream_edges_assign)[curr_node_id][p]) {
                edge_load_p[0]++;
            }
            if(!(*config.stream_edges_assign)[neighbour_node.first][p]) {
                edge_load_p[0]++;
            }
            edge_load_p[1] = neighbour_node.second;

			if (blocks[p].fully_loaded_edge_multi_constraints(edge_load_p)) {
				continue;
			}

			gu = 0, gv = 0, gu_c = 0, gv_c = 0;
			sum = degree_curr_node + degree_neighbour_node;
			//sum_of_volumes = volumes[communities[e.first]] + volumes[communities[e.second]];
            sum_of_volumes = (*config.stream_blocks_weight)[curr_node_id] + (*config.stream_blocks_weight)[neighbour_node.first];
            
			if ((*config.stream_edges_assign)[curr_node_id][p]) {
				gu = degree_curr_node;
				gu /= sum;
				gu = 1 + (1-gu);
				if ((*config.stream_nodes_assign)[curr_node_id] == p){
					gu_c = (*config.stream_blocks_weight)[curr_node_id];
					gu_c /= sum_of_volumes;
				}
			}

			if ((*config.stream_edges_assign)[neighbour_node.first][p]) {
				gv = degree_neighbour_node;
				gv /= sum;
				gv = 1 + (1-gv);
				if ((*config.stream_nodes_assign)[neighbour_node.first] == p){
					gv_c = (*config.stream_blocks_weight)[neighbour_node.first];
					gv_c /= sum_of_volumes;
				}
			}

			double score_p = gu + gv + gu_c + gv_c;
			if (score_p < 0) {
				std::cout << "ERROR: score_p < 0" << std::endl;
				std::cout << "gu: " << gu << std::endl;
				std::cout << "gv: " << gv << std::endl;
				std::cout << "gu_c: " << gu_c << std::endl;
				std::cout << "gv_c: " << gv_c << std::endl;
				std::cout << "bal: " << bal << std::endl;
				exit(-1);
			}

			if (score_p >= max_score) {
				max_score = score_p;
				max_p = p;
			}
		}
    }
    return max_p;
}

void onepass_linear_obj_edge::instantiate_blocks(LongNodeID n, LongEdgeID m,
                                        PartitionID k, PartitionID number_of_constraints, ImbalanceType epsilon, ImbalanceType epsilon_edge, HeiClus::PartitionConfig & config) {
    vertex_partitioning::instantiate_blocks(n, m, k, number_of_constraints, epsilon, epsilon_edge, config);
    auto &blocks = vertex_partitioning::blocks;
    for (auto &block : blocks) {
        block.set_fennel_constants(n, m, k, this->gamma);
    }
}
