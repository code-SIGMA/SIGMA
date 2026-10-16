/******************************************************************************
 * hdrf.cpp
 * *
 * Source of KaHIP -- Karlsruhe High Quality Partitioning.
 *****************************************************************************/

#include "hdrf.h"
#include "partition/onepass_partitioning/floating_block.h"
#include "partition/onepass_partitioning/vertex_partitioning.h"
#include <cmath>

onepass_hdrf::onepass_hdrf(PartitionID k0, PartitionID kf,
                               PartitionID max_blocks, NodeID n_threads,
                               bool hashing /*=false*/, float gamma /*=1.5*/)
    : vertex_partitioning(k0, kf, max_blocks, n_threads, hashing) {
    this->gamma = gamma;
}

onepass_hdrf::~onepass_hdrf() {}

double onepass_hdrf::compute_score(floating_block &block, int my_thread, HeiClus::PartitionConfig &config, PartitionID cluster, int restreaming, LongNodeID curr_node_id, std::vector<floating_block> & blocks, std::pair<NodeID, EdgeWeight> const& neighbour_node) {
    
	// e.first = curr_node_id; e.second = neighbour_node.first

	EdgeWeight degree_u = (*config.node_degrees)[curr_node_id];
	EdgeWeight degree_v = (*config.node_degrees)[neighbour_node.first];

	uint32_t sum;
	double max_score = 0;
	uint32_t max_p = -1;
	double gv, gu;

	double bal_edge, bal_node;

	const double epsilon = 1;

	std::vector<NodeWeight> curr_edge_weight;
	curr_edge_weight.resize(config.number_of_constraints);

	const double mult_dyn = config.mult_capacity_y
		+ (1.0 - config.mult_capacity_y) * config.stream_progress;

	const double capacity_scale = config.use_capacity_curve
		? config.capacity_min_cap + (1.0 - config.capacity_min_cap)
			  * std::pow(config.stream_progress, config.capacity_alpha)
		: 1.0;

	for (auto & block : blocks) {
		PartitionID current_block_id = block.get_block_id();

		curr_edge_weight[0] = 0;
    	curr_edge_weight[1] = 0;

		// Adjust the weights associated with assigning edge to node_partition
		if(!(*config.stream_edges_assign)[curr_node_id][current_block_id]) {
			curr_edge_weight[0]++;
		}
		if(!(*config.stream_edges_assign)[neighbour_node.first][current_block_id]) {
			curr_edge_weight[0]++;
		}

		curr_edge_weight[1] = neighbour_node.second;

		if (blocks[current_block_id].fully_loaded_edge_multi_constraints(curr_edge_weight, mult_dyn, capacity_scale)) {
			// Partition already fully loaded so dont pre-assign edge
			continue;
		}

		gu = 0, gv = 0;
		sum = degree_u + degree_v;
		if ((*config.stream_edges_assign)[curr_node_id][current_block_id]) {
			gu = degree_u;
			gu/=sum;
			gu = 1 + (1-gu);
		}
		if ((*config.stream_edges_assign)[neighbour_node.first][current_block_id]) {
			gv = degree_v;
			gv /= sum;
			gv = 1 + (1-gv);
		}
		
		// node balance and edge balance needs to be taken into account.
		// bal = max_load - edge_load[p];
		// if (min_load != UINT64_MAX) bal /= epsilon + max_load - min_load;
		// double score_p = gu + gv + globals.LAMBDA*bal;
		
		bal_edge = blocks[config.max_load_edges_part_hdrf].partition_weights[1] - blocks[current_block_id].partition_weights[1];
		bal_edge /= epsilon + blocks[config.max_load_edges_part_hdrf].partition_weights[1] - 1;

		bal_node = blocks[config.max_load_nodes_part_hdrf].partition_weights[0] - blocks[current_block_id].partition_weights[0];
		bal_node /= epsilon + blocks[config.max_load_nodes_part_hdrf].partition_weights[0] - 1;

		double score_p = gu + gv + config.lambda_hdrf * ( config.alpha_hdrf * bal_edge + (1.0 - config.alpha_hdrf) * bal_node );

		if (score_p < 0) {
			std::cout << "ERROR: score_p < 0" << std::endl;
			std::cout << "gu: " << gu << std::endl;;
			std::cout << "gv: " << gv << std::endl;
			exit(-1);
		}
		if (score_p > max_score) {
			max_score = score_p;
			max_p = current_block_id;
		}
	}
	return max_p;
}

void onepass_hdrf::instantiate_blocks(LongNodeID n, LongEdgeID m,
                                        PartitionID k, PartitionID number_of_constraints, ImbalanceType epsilon, ImbalanceType epsilon_edge, HeiClus::PartitionConfig & config) {
    vertex_partitioning::instantiate_blocks(n, m, k, number_of_constraints, epsilon, epsilon_edge, config);
    auto &blocks = vertex_partitioning::blocks;
    for (auto &block : blocks) {
        block.set_fennel_constants(n, m, k, this->gamma);
    }
}
