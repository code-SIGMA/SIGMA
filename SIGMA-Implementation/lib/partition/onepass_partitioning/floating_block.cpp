/******************************************************************************
 * floating_block.h 
 * *
 * Source of KaHIP -- Karlsruhe High Quality Partitioning.
 * Marcelo Fonseca Faraj
 *****************************************************************************/


#include "floating_block.h"
#include "partition/onepass_partitioning/vertex_partitioning.h"


floating_block::floating_block(vertex_partitioning* parent_problem, PartitionID my_block_id, NodeID n_threads, PartitionID number_of_constraints, double fennel_gamma, PartitionID depth/*=0*/ )
	:  partition_weights(number_of_constraints, 0),
	   partition_constraints(number_of_constraints, 0)
{
	this->my_block_id = my_block_id;
	current_weight = 0;
	alpha = 1;
	gamma = fennel_gamma;
	alpha_gamma = alpha * gamma;
	multiplicative_constant = 1;
	
	e_weight = 0;
	sigma_cluster = 0;
}

void floating_block::set_capacity(LongNodeID capacity) {
	this->capacity = capacity;
}

void floating_block::set_mult_capacity(std::vector<ImbalanceType> & mult_capacity) {
	this->partition_constraints = mult_capacity;
}

void floating_block::set_fennel_constants(float n, float m, float k, float gamma) {
	this->gamma = gamma;
	this->alpha = m * std::pow(k,gamma-1) / (std::pow(n,gamma));
	this->alpha_gamma = this->alpha * this->gamma;
}

// Inlined into header:
// fully_loaded_multi_constraints
// fully_loaded_edge_multi_constraints

float floating_block::get_block_id() const {
	return my_block_id;
}

float floating_block::get_fennel_obj() {
	float result = (0.1+(e_weight))-multiplicative_constant * (alpha_gamma * std::pow(current_weight , gamma-1) );
	return result;
}

float floating_block::get_fennel_approx_sqrt_obj() {
	float result = (0.1+(e_weight))-multiplicative_constant * (alpha_gamma*HeiClus::random_functions::approx_sqrt(current_weight) );
	return result;
}

// modified fennel that runs between -1 and 1 (what is currently being used)

float floating_block::get_fennel_range_obj(EdgeWeight curr_node_degree) {
	//Calculate Penalty

    // Compute max components
    float max = 0.0f;
    for (size_t i = 0; i < partition_weights.size(); ++i) {
        float v = static_cast<float>(partition_weights[i]) / static_cast<float>(partition_constraints[i]);
        if (v > max) max = v;
    }
	
	float result = -1.0f;
	if(curr_node_degree == 0) {
		// result = 0 -  ( alpha_gamma *std::pow(max , gamma - 1.1) ); // Only consider balance.
		result = 0 - ( std::pow(max , gamma - 1.1f) ); //remove alpha gamma
	} else {
		// result = (e_weight / curr_node_degree) -  ( alpha_gamma *std::pow(max , gamma - 1.1) );
		result = (e_weight / curr_node_degree) - ( std::pow(max , gamma - 1.1f) );
	}
    return result;
}


float floating_block::get_fennel_multi_obj(NodeID curr_node_id, EdgeWeight curr_node_degree, std::vector<LongNodeID> & curr_node_neighbours, std::vector<PartitionID> & stream_nodes_assign, EdgeAssignmentArray & stream_edges_assign, double tau_mult_obj, PartitionID k) {
	
	//Calculate Penalty

    // Compute max components
    float max = 0.0f;
    for (size_t i = 0; i < partition_weights.size(); ++i) {
        float v = static_cast<float>(partition_weights[i]) / static_cast<float>(partition_constraints[i]);
        if (v > max) max = v;
    }

	// Replication factor penalty:
	// if node "curr_node_id" is put in block "my_block_id", then for every neighbour "neighbour"

	ImbalanceType replication_factor_pen = 0;
	NodeID num_neighbour_nodes_replicated = 0;
	NodeID num_current_node_replicated = 0;

	std::bitset<MAX_NUM_PARTITION> part_current_node_replicated;
	part_current_node_replicated.reset();

	for(auto & neighbour : curr_node_neighbours) {

		// "neighbour" will need to be replicated in block my_block_id, unless stream_edges_assign[neighbour][my_block_id] is already true
		if (!stream_edges_assign[neighbour][my_block_id]) {
			num_neighbour_nodes_replicated++;
		}

		// "curr_node_id" will need to be replicated in the block Vi that "neigbour" was assigned to (stream_nodes_assign[neighbour] = Vi), 
		// unless stream_edges_assign[curr_node_id][Vi] is already true
		if(stream_nodes_assign[neighbour] != -1 
			&& stream_nodes_assign[neighbour] != my_block_id
			&& (!stream_edges_assign[curr_node_id][stream_nodes_assign[neighbour]])) {
			part_current_node_replicated[stream_nodes_assign[neighbour]] = true;
		}
	}

	num_current_node_replicated = part_current_node_replicated.count();

	replication_factor_pen = (num_current_node_replicated + num_neighbour_nodes_replicated);
	
	float result = -1.0f;
	if(curr_node_degree == 0) {
		result = 0 - std::pow(max , gamma - 1.1f) - ( replication_factor_pen / ( curr_node_degree * 2 ) );
	} else {
		result = (e_weight / curr_node_degree) -  std::pow(max , gamma - 1.1f) - ( tau_mult_obj * ( replication_factor_pen / ( curr_node_degree + k ) ) );
	}
    return result;
}

float floating_block::get_fennel_multi_obj_fast(
    NodeID degree_count, EdgeWeight curr_node_degree,
    const std::vector<NodeID>& neighbor_rep_count,
    NodeID pre_rep_count, bool my_block_in_full_rep_set,
    double tau_mult_obj, PartitionID k) {
    
	// Balance penalty - same computation as get_fennel_multi_obj
    float max = 0.0f;
    const auto* pw = partition_weights.data();
    const auto* pc = partition_constraints.data();
	
    for (size_t i = 0; i < partition_weights.size(); ++i) {
        float v = static_cast<float>(pw[i]) / static_cast<float>(pc[i]);
        if (v > max) max = v;
    }

    // Replication penalty (O(1), precomputed once per node):
    // - num_neighbour_nodes_replicated: neighbours not yet in my_block_id (need pulling in).
    // - num_current_node_replicated: blocks where curr_node would need to be replicated
    //   (blocks holding a neighbour where curr_node is absent); subtract 1 if my_block_id
    //   is one of them, since assigning here removes that replication need.
    NodeID num_neighbour_nodes_replicated = degree_count - neighbor_rep_count[my_block_id];
    NodeID num_current_node_replicated    = pre_rep_count - (my_block_in_full_rep_set ? 1u : 0u);
    ImbalanceType pen = num_neighbour_nodes_replicated + num_current_node_replicated;

    if (curr_node_degree == 0) {
        return -std::pow(max, gamma - 1.1f);
    }
    return (e_weight / curr_node_degree)
         - std::pow(max, gamma - 1.1f)
         - (float)(tau_mult_obj * (pen / (curr_node_degree + k)));
}


void floating_block::increment_curr_mult_weight(const std::vector<NodeWeight>& weight) {
    const size_t num_weights = partition_weights.size();
    auto* const p_weights = partition_weights.data();
    const auto* const p_in = weight.data();

	for(size_t i = 0; i < num_weights; ++i) {
		p_weights[i] += p_in[i];
	}
}