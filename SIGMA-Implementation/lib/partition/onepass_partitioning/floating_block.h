/******************************************************************************
 * floating_block.h 
 * *
 * Source of KaHIP -- Karlsruhe High Quality Partitioning.
 * Marcelo Fonseca Faraj
 *****************************************************************************/

#ifndef FLOATINGBLOCKS_7I4IR31Y
#define FLOATINGBLOCKS_7I4IR31Y

#include <algorithm>
#include <vector>
#include <omp.h>

#include "random_functions.h"
#include "timer.h"
#include "definitions.h"

#include "partition/partition_config.h"

class vertex_partitioning;

class floating_block {
        public:
                floating_block(vertex_partitioning* parent_problem, PartitionID my_block_id, NodeID n_threads, PartitionID number_of_constraints, double fennel_gamma, PartitionID depth=0);
                ~floating_block(){}
		void increment_edgeweight(int my_thread, EdgeWeight e_weight);
		void set_edgeweight(int my_thread, EdgeWeight e_weight);
		void update_cluster_sigma(LongEdgeID degree);

		EdgeWeight get_edgeweight(int my_thread) const;
		float get_leiden_obj(int my_thread, float cpm_gamma, HeiClus::PartitionConfig &config, PartitionID cluster);
		double get_modularity_obj(int my_thread, HeiClus::PartitionConfig &config, EdgeWeight current_graph_edge_count, EdgeWeight streamed_curr_node_edge_count);
		double get_modularity_restream(int my_thread, HeiClus::PartitionConfig &config, EdgeWeight current_graph_edge_count, EdgeWeight streamed_curr_node_edge_count, LongNodeID curr_node_id, std::vector<floating_block> & blocks);
		double get_modularity_max_cluster(int my_thread, HeiClus::PartitionConfig &config, EdgeWeight current_graph_edge_count, EdgeWeight streamed_curr_node_edge_count, LongNodeID curr_node_id, std::vector<floating_block> & blocks);

		float get_fennel_obj();
		float get_fennel_approx_sqrt_obj();
		
		float get_fennel_multi_obj(NodeID curr_node_id,EdgeWeight curr_node_degree, std::vector<LongNodeID> & curr_node_neighbours, std::vector<PartitionID> & stream_nodes_assign, EdgeAssignmentArray & stream_edges_assign, double tau_mult_obj, PartitionID k);
		
		float get_fennel_multi_obj_fast(NodeID degree_count, EdgeWeight curr_node_degree,
		                                const std::vector<NodeID>& neighbor_rep_count,
		                                NodeID pre_rep_count, bool my_block_in_full_rep_set,
		                                double tau_mult_obj, PartitionID k);
		float get_fennel_range_obj(EdgeWeight curr_node_degree);
		
		bool fully_loaded_multi_constraints(const std::vector<NodeWeight> & curr_node_weight, double mult = 1.0, double capacity_scale = 1.0) const;
		bool fully_loaded_edge_multi_constraints(const std::vector<NodeWeight> & curr_edge_weight, double mult = 1.0, double capacity_scale = 1.0) const;

		float get_block_id() const;

		void set_capacity(LongNodeID capacity);
		void set_mult_capacity(std::vector<ImbalanceType> & mult_capacity);
		void set_fennel_constants(float n, float m, float k, float gamma);

		void increment_curr_mult_weight(const std::vector<NodeWeight>& n_weight);

	/* private: */
		NodeID sigma_cluster;
		EdgeWeight e_weight;
		
		LongNodeID capacity;

		PartitionID my_block_id;
		
		// For Fennel
		float multiplicative_constant;
		float alpha_gamma;
		float gamma;
		float alpha;

		NodeWeight current_weight;

		// multiple weights
		std::vector<ImbalanceType> partition_constraints;
		std::vector<ImbalanceType> partition_weights;
};

inline void floating_block::increment_edgeweight(int my_thread, EdgeWeight e_weight) {
	this->e_weight += e_weight;
}

inline void floating_block::set_edgeweight(int my_thread, EdgeWeight e_weight) {
	this->e_weight = e_weight;
}

inline void floating_block::update_cluster_sigma(LongEdgeID degree) {
	this->sigma_cluster += degree;
}

inline EdgeWeight floating_block::get_edgeweight(int my_thread) const {
	return e_weight;
}

inline float floating_block::get_leiden_obj(int my_thread, float cpm_gamma, HeiClus::PartitionConfig &config, PartitionID cluster) {
	// delta CPM
	double result;
	result = -cpm_gamma * (*config.stream_blocks_weight)[cluster] + e_weight;
	result *= 2;
	return result;
}

inline double floating_block::get_modularity_obj(int my_thread, HeiClus::PartitionConfig &config, EdgeWeight current_graph_edge_count, EdgeWeight streamed_curr_node_edge_count)  {
	
	// Delta modularity with streamed node being assigned its own cluster.
	double first_term = ((1.0 / static_cast<double>(config.total_edges)) * e_weight);
	double second_term = (((static_cast<double>(streamed_curr_node_edge_count)/2.0) / static_cast<double>(config.total_edges))/static_cast<double>(config.total_edges)) * (static_cast<double>(sigma_cluster));
	double result = first_term - second_term;
	return result;
}

inline double floating_block::get_modularity_restream(int my_thread, HeiClus::PartitionConfig &config, EdgeWeight current_graph_edge_count, EdgeWeight streamed_curr_node_edge_count, LongNodeID curr_node_id, std::vector<floating_block> & blocks) {
	
	// Delta modularity with streamed node already assigned to a cluster. Currently in restream phase. Streamed node cluster volume to be considered.
	PartitionID orig_part = (*config.stream_nodes_assign)[curr_node_id];
	
	double first_term = ((1.0 / static_cast<double>(config.total_edges)) * (e_weight - blocks[orig_part].e_weight));
	double second_term = (((static_cast<double>(streamed_curr_node_edge_count)/2.0) / static_cast<double>(config.total_edges))/static_cast<double>(config.total_edges)) * (static_cast<double>(streamed_curr_node_edge_count) + static_cast<double>(sigma_cluster) - static_cast<double>(blocks[orig_part].sigma_cluster));
	double result = first_term - second_term;
	return result;
}

inline double floating_block::get_modularity_max_cluster(int my_thread, HeiClus::PartitionConfig &config, EdgeWeight current_graph_edge_count, EdgeWeight streamed_curr_node_edge_count, LongNodeID curr_node_id, std::vector<floating_block> & blocks) {
	
	// Delta modularity where max number of clusters has been reached. Assign node to max cluster ID.
	PartitionID orig_part = blocks.size()-1;
	
	double first_term = ((1.0 / static_cast<double>(config.total_edges)) * (e_weight - blocks[orig_part].e_weight));
	double second_term = (((static_cast<double>(streamed_curr_node_edge_count)/2.0) / static_cast<double>(config.total_edges))/static_cast<double>(config.total_edges)) * (static_cast<double>(streamed_curr_node_edge_count) + static_cast<double>(sigma_cluster) - static_cast<double>(blocks[orig_part].sigma_cluster));
	double result = first_term - second_term;
	return result;
}

inline bool floating_block::fully_loaded_multi_constraints(const std::vector<NodeWeight> & curr_node_weight, double mult, double capacity_scale) const {
    const size_t num_constraints = partition_weights.size();
    const auto* const p_weights = partition_weights.data();
    const auto* const p_constraints = partition_constraints.data();

	for(size_t i = 0; i < num_constraints; ++i) {
		if(static_cast<double>(p_weights[i]) + mult * static_cast<double>(curr_node_weight[i]) > capacity_scale * static_cast<double>(p_constraints[i])) {
			return true;
		}
	}
	return false;
}

inline bool floating_block::fully_loaded_edge_multi_constraints(const std::vector<NodeWeight> & curr_edge_weight, double mult, double capacity_scale) const {
    const size_t num_constraints = partition_weights.size();
    const auto* const p_weights = partition_weights.data();
    const auto* const p_constraints = partition_constraints.data();
    const auto* const p_curr = curr_edge_weight.data();

	for(size_t i = 0; i < num_constraints; ++i) {
		if(static_cast<double>(p_weights[i]) + mult * static_cast<double>(p_curr[i]) > capacity_scale * static_cast<double>(p_constraints[i])) {
			return true;
		}
	}
	return false;
}

#endif /* end of include guard: FLOATINGBLOCKS_7I4IR31Y */