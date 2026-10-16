/******************************************************************************
 * vertex_partitioning.cpp
 * *
 * Source of KaHIP -- Karlsruhe High Quality Partitioning.
 * Marcelo Fonseca Faraj
 *****************************************************************************/

#include "vertex_partitioning.h"
#include "partition/onepass_partitioning/floating_block.h"
#include <cmath>

vertex_partitioning::vertex_partitioning(PartitionID k0, PartitionID kf,
                                         PartitionID max_blocks,
                                         NodeID n_threads,
                                         int mode,
                                         bool hashing /*=false*/) {
  this->k0 = k0;
  this->kf = kf;
  this->max_blocks = max_blocks;
  this->n_threads = n_threads;
  this->neighbor_blocks.resize(n_threads);
  original_problem = NULL;
  subproblem_tree_root = NULL;
  parent_block = NULL;
  this->hashing = hashing;
  this->use_self_sorting_array = false;
  int largest_dim = max_blocks;
  EdgeWeight current_graph_edge_count = 0;
  EdgeWeight streamed_curr_node_edge_count = 0;
  this->quotient_edge_count = 0;
  this->overall_delta_mod = 0;
  if(mode != LIGHT_PLUS && mode != LIGHT ) {
    //quotient = new robin_hood::unordered_flat_map<std::pair<PartitionID, PartitionID>, EdgeWeight, PairHash>();
    quotient = new absl::flat_hash_map<std::pair<PartitionID, PartitionID>, EdgeWeight, PairHash>();

  }
}


double vertex_partitioning::compute_score(floating_block &block, int my_thread, HeiClus::PartitionConfig &config, PartitionID cluster, int restreaming, LongNodeID curr_node_id, std::vector<floating_block> & blocks, std::pair<NodeID, EdgeWeight> const& neighbour_node) {
  return 0;
}

void vertex_partitioning::update_quotient_graph(PartitionID block,
                                                std::vector<std::pair<PartitionID, EdgeWeight>> &neighbor_blocks) {

  // Loop over the neighbours of the streamed node to update / insert edges in the quotient Graph.                              
  for (auto &neighbour_block : neighbor_blocks) {
    if(neighbour_block.first == -1) {
      break;
    }

    PartitionID neighbour = neighbour_block.first;
    EdgeWeight weight = neighbour_block.second;

    std::pair<PartitionID, PartitionID> key = (block <= neighbour) 
             ? std::make_pair(block, neighbour) 
             : std::make_pair(neighbour, block);
             
     // Insert or update the entry in the quotient graph
     EdgeWeight &current_weight = (*quotient)[key];
     if (current_weight == 0 && block != neighbour) {
        // Increment edge count only for new edges (excluding self-loops)
         quotient_edge_count++;
     }
     current_weight += weight;
  }
}

void vertex_partitioning::instantiate_blocks(LongNodeID n, LongEdgeID m,
                                             PartitionID k,
                                             PartitionID number_of_constraints,
                                             ImbalanceType epsilon,
                                             ImbalanceType epsilon_edge,
                                             HeiClus::PartitionConfig & config ) {
  if (blocks.size() > 0)
    return;
  
  base_size_constraint = ceil(((100 + epsilon) / 100.) * (n / (double)k));
  
  mult_constraints.resize(number_of_constraints, 0);
  
  // Edge and Node Balance
  if(config.partitioning_type == NODE_PARTITIONING) { 
    mult_constraints[0] = ceil(((100 + epsilon) / 100.) * (n / (double)k));
    mult_constraints[1] = ceil(((100 + epsilon_edge) / 100.) * ((2*m + n) / (double)k));
  } else {
    mult_constraints[0] = INFINITY;
    mult_constraints[1] = ceil(((100 + epsilon_edge) / 100.) * ((m) / (double)k));
  }
  
  for (PartitionID id = 0; id < k; id++) {
    blocks.push_back(floating_block(this, id, n_threads, number_of_constraints, config.fennel_gamma));
    blocks[id].set_mult_capacity(mult_constraints);
    blocks[id].set_capacity(base_size_constraint);
  }
  
  if (this->use_self_sorting_array == true) {
    this->sorted_blocks.initialize(k, (NodeWeight)0);
  }
}

void vertex_partitioning::adjust_capacity( NodeID curr_node_id, HeiClus::PartitionConfig & config) {
  
  NodeID num_assigned_nodes = 0;
  NodeID num_assigned_edges = 0;

  for(PartitionID part = 0; part < config.k; part++) {
      num_assigned_nodes += (*config.stream_blocks_load)[part][0];
      num_assigned_edges += (*config.stream_blocks_load)[part][1];
  }

  if(config.partitioning_type == NODE_PARTITIONING) {
    if((config.total_nodes - num_assigned_nodes) > ( config.total_nodes / config.num_capacity_changes ) ) {
      // mult_constraints[0] = ceil(((100 + config.epsilon) / 100.) * ( ( num_assigned_nodes + ( config.total_nodes / config.num_capacity_changes ) ) / (double)config.k));
      mult_constraints[0] = ceil(( num_assigned_nodes + ( config.total_nodes / config.num_capacity_changes ) ) / (double)config.k);
    } else {
      mult_constraints[0] = ceil(((100 + config.epsilon) / 100.) * (config.total_nodes / (double)config.k));
    }
  } else {
    mult_constraints[0] = INFINITY;
  }
  
  if(config.partitioning_type == NODE_PARTITIONING) {
    if(((2 * config.total_edges + config.total_nodes) - num_assigned_edges ) > ( ( 2 * config.total_edges + config.total_nodes ) / config.num_capacity_changes )) {
      // mult_constraints[1] = ceil(((100 + config.epsilon_edge) / 100.) * ( ( ( 2 * num_assigned_edges + num_assigned_nodes ) + ( ( 2 * config.total_nodes + config.total_edges )  / config.num_capacity_changes ) )  / (double)config.k));
      mult_constraints[1] = ceil(( num_assigned_edges + ( ( 2 * config.total_edges + config.total_nodes )  / config.num_capacity_changes ) )  / (double)config.k) ;
    } else {
      mult_constraints[1] = ceil(((100 + config.epsilon_edge) / 100.) * ((2 * config.total_edges + config.total_nodes) / (double)config.k));
    }
  } else {
    if((config.total_edges - num_assigned_edges) > (config.total_edges / config.num_capacity_changes )) {
      mult_constraints[1] = ceil(( num_assigned_edges + ( ( 2 * config.total_edges + config.total_nodes )  / config.num_capacity_changes ) )  / (double)config.k) ;
    } else {
      mult_constraints[1] = ceil(((100 + config.epsilon_edge) / 100.) * ((config.remaining_stream_edges) / (double)config.k));
    }
  }
  

  for (PartitionID id = 0; id < config.k; id++) {
    blocks[id].set_mult_capacity(mult_constraints);
  }
  
}

void vertex_partitioning::update_initial_blocks_capacity( HeiClus::PartitionConfig & config) {
  
  NodeID num_preassigned_nodes = 0;
  NodeID num_preassigned_edges = 0;

  for(PartitionID part = 0; part < config.k; part++) {
      num_preassigned_nodes += (*config.stream_blocks_load)[part][0];
      num_preassigned_edges += (*config.stream_blocks_load)[part][1];
  }

  if(config.total_nodes - num_preassigned_nodes > ( config.total_nodes / config.num_capacity_changes ) ) {
    mult_constraints[0] = ceil(((100 + config.epsilon) / 100.) * ( ( num_preassigned_nodes + ( config.total_nodes / config.num_capacity_changes ) ) / (double)config.k));
  } else {
    mult_constraints[0] = ceil(((100 + config.epsilon) / 100.) * (config.total_nodes / (double)config.k));
  }

  if((2 * config.total_edges + config.total_nodes) - ( 2 * num_preassigned_edges + num_preassigned_nodes ) > ( ( 2 * config.total_nodes + config.total_edges ) / config.num_capacity_changes )) {
    mult_constraints[1] = ceil(((100 + config.epsilon_edge) / 100.) * ( ( ( 2 * num_preassigned_edges + num_preassigned_nodes ) + ( ( 2 * config.total_nodes + config.total_edges )  / config.num_capacity_changes ) )  / (double)config.k));
  } else {
    mult_constraints[1] = ceil(((100 + config.epsilon_edge) / 100.) * ((2*config.total_edges + config.total_nodes) / (double)config.k));
  }
  

  for (PartitionID id = 0; id < config.k; id++) {
    blocks[id].set_mult_capacity(mult_constraints);
  }
}

PartitionID vertex_partitioning::assign_to_partition( LongNodeID curr_node_id, 
                                                      std::vector<NodeWeight> & curr_node_weight,
                                                      PartitionID previous_assignment,
                                                      HeiClus::PartitionConfig & config, 
                                                      double kappa,
                                                      int my_thread,
                                                      std::ofstream & part_edge_file) {
  
  if(config.partitioning_type == NODE_PARTITIONING) {
      PartitionID block = solve_node_partitioning(curr_node_id, curr_node_weight, previous_assignment, config, config.kappa, my_thread);
      return block;
  } else {
      PartitionID block = solve_edge_partitioning(curr_node_id, curr_node_weight, previous_assignment, config, config.kappa, my_thread, part_edge_file);
      return block;
  }

}

PartitionID vertex_partitioning::solve_node_partitioning( LongNodeID curr_node_id, 
                                                          std::vector<NodeWeight> & curr_node_weight,
                                                          PartitionID previous_assignment,
                                                          HeiClus::PartitionConfig & config, 
                                                          double kappa,
                                                          int my_thread) {
  
  float best = std::numeric_limits<float>::lowest();
  float score;
  PartitionID k = blocks.size();
  PartitionID decision = -1;
  bool valid_part_found = false;

  const double mult_dyn = config.mult_capacity_y
      + (1.0 - config.mult_capacity_y) * config.stream_progress;

  const double capacity_scale = config.use_capacity_curve
      ? config.capacity_min_cap + (1.0 - config.capacity_min_cap)
            * std::pow(config.stream_progress, config.capacity_alpha)
      : 1.0;

  // Allow onepass_fennel_multi_obj to precompute per-node data once before iterating over all k blocks.
  precompute_for_node(curr_node_id, config);

  /* PartitionID decision = random_functions::nextInt(0, k-1); */

  for (auto &block : blocks) {
    if (block.fully_loaded_multi_constraints(curr_node_weight, mult_dyn, capacity_scale)) {
      continue;
    }

    valid_part_found = true;

    score = compute_score(block, my_thread, config, block.get_block_id(), 0, curr_node_id, blocks, std::make_pair(-1, -1));
    // float old_score = score;
    if (block.get_block_id() == previous_assignment) {
      if (score > 0) {
        score = score * kappa; // mult not good enough
      } else {
        score = score / kappa;
      }
    }

    if ((score > best || ( score == best && block.get_block_id() < decision )) && config.one_pass_algorithm == ONEPASS_FENNEL_MULT_OBJ ) {
      // Tie breaks: assign to the lowest partition ID, for fennel mult obj
      decision = block.get_block_id();
      best = score;
    } else if ((score > best || (random_obj.nextBool() && score == best)) && config.one_pass_algorithm != ONEPASS_FENNEL_MULT_OBJ) {
      decision = block.get_block_id();
      best = score;
    }
  }

  if(!valid_part_found) {
    std::vector<float> part_load(config.number_of_constraints);
    EdgeWeight min_load = std::numeric_limits<EdgeWeight>::infinity();
    PartitionID min_p = 0;
    
    for (PartitionID i = 0; i < config.k; i++){
      part_load[0] = static_cast<float>(blocks[i].partition_weights[0]) / static_cast<float>(blocks[i].partition_constraints[0]);
      part_load[1] = static_cast<float>(blocks[i].partition_weights[1]) / static_cast<float>(blocks[i].partition_constraints[1]);

      float max_overload = 0.0f;
      if(part_load[0] > part_load[1]) {
        max_overload = part_load[0];
      } else {
        max_overload = part_load[1];
      }

      if (max_overload < min_load){
    	  min_load = max_overload;
    		min_p = i;
    	}
		}

    decision = min_p;
  }

  // When using the fennel multi objective scoring function we also need to update the stream_edges_assign.
  if(config.one_pass_algorithm == ONEPASS_FENNEL_MULT_OBJ) {
    (*config.stream_edges_assign)[curr_node_id][decision] = true;

    for(auto & neighbour : config.curr_node_neighbours) {
      (*config.stream_edges_assign)[neighbour][decision] = true;
    }
  }

  return set_decision_partitioning(decision, curr_node_id, curr_node_weight, my_thread);
}

PartitionID vertex_partitioning::solve_edge_partitioning( LongNodeID curr_node_id, 
                                                          std::vector<NodeWeight> & curr_node_weight,
                                                          PartitionID previous_assignment,
                                                          HeiClus::PartitionConfig & config, 
                                                          double kappa,
                                                          int my_thread,
                                                          std::ofstream & part_edge_file) {
  
  std::vector<NodeWeight> curr_edge_weight;
	curr_edge_weight.resize(config.number_of_constraints); 

  PartitionID max_partition = -1;

  for (auto &neighbour : config.neighbor_blocks[my_thread]) {

    if(neighbour.first == -1) {
      break;
    }

    if(curr_node_id > neighbour.first) {
      continue; // edge has already been assigned
    }

    // Check if edge has been preassigned
    PartitionID partition_u = (*config.stream_nodes_assign)[curr_node_id];
	  PartitionID partition_v = (*config.stream_nodes_assign)[neighbour.first];
	  if (partition_u == partition_v) {
      
      if(!config.suppress_file_output) {
        part_edge_file << curr_node_id << " " << neighbour.first<< " " << partition_u << '\n';
      }
      
      continue;
    }

    max_partition = compute_score(blocks[0], my_thread, config, -1, 0, curr_node_id, blocks, neighbour);

    if(max_partition == -1) {
      std::vector<float> part_load(config.number_of_constraints);
      EdgeWeight min_load = std::numeric_limits<EdgeWeight>::infinity();
      PartitionID min_p = 0;

      // TODO: get partition assign edge to partition which is least loaded.
      for (PartitionID i = 0; i < config.k; i++){
        //part_load[0] = static_cast<float>(blocks[i].partition_weights[0]) / static_cast<float>(blocks[i].partition_constraints[0]);
        part_load[1] = static_cast<float>(blocks[i].partition_weights[1]) / static_cast<float>(blocks[i].partition_constraints[1]);

        // float average_load = part_load[0] + part_load[1];
        // average_load /= config.number_of_constraints;
        
        if (part_load[1] < min_load){
          min_load = part_load[1];
          min_p = i;
        }
      }
      max_partition = min_p;
    }
      
    curr_edge_weight[0] = 0;
    curr_edge_weight[1] = 0;

    // Adjust the weights associated with assigning edge to node_partition 
    if(!(*config.stream_edges_assign)[curr_node_id][max_partition]) {
      curr_edge_weight[0]++;
    }
    if(!(*config.stream_edges_assign)[neighbour.first][max_partition]) {
      curr_edge_weight[0]++;
    }

    curr_edge_weight[1] = neighbour.second;

    // Set Edge Partitioning decision.
    max_partition = set_decision_edge_partitioning(max_partition, curr_node_id, curr_edge_weight);

    // Assign the endpoints to the partition. (update_vertex_partition_matrix(e, max_p))
    (*config.stream_edges_assign)[curr_node_id][max_partition] = true;
    (*config.stream_edges_assign)[neighbour.first][max_partition] = true;
      
    (*config.stream_blocks_weight)[max_partition] += curr_edge_weight[0];
      
    if(blocks[max_partition].partition_weights[0] > blocks[config.max_load_nodes_part_hdrf].partition_weights[0]) {
      config.max_load_nodes_part_hdrf = max_partition;
    }

    if(blocks[max_partition].partition_weights[1] > blocks[config.max_load_edges_part_hdrf].partition_weights[1]) {
      config.max_load_edges_part_hdrf = max_partition;
    }

    for(int i = 0; i < config.number_of_constraints; i++) {
      (*config.stream_blocks_load)[max_partition][i] += curr_edge_weight[i];
    }

    if(!config.suppress_file_output) {
      part_edge_file << curr_node_id << " " << neighbour.first<< " " << max_partition << '\n';
    }
  }  

  // Note if execution did not enter for loop, node is empty and has no edges, so no edges also need to be assigned

  return max_partition;
}


PartitionID vertex_partitioning::pre_assign_node( LongNodeID curr_node_id, 
                                                  std::vector<NodeWeight> & curr_node_weight,
                                                  std::vector<std::pair<PartitionID, EdgeWeight>> &neighbours,
                                                  HeiClus::PartitionConfig &config,
                                                  std::vector<PartitionID> & com2part,
                                                  std::vector<PartitionID> & cluster_assignment) {
  
  PartitionID curr_cluster = cluster_assignment[curr_node_id];
  PartitionID node_partition = com2part[curr_cluster];
  
  for (PartitionID key = 0; key < config.next_key[0]; key++) {
    auto & neighbour = neighbours[key];
    
    if(node_partition != neighbour.first) {
      // not all neighbours have the same assignmenet as current node --> node is an edge node.
      return node_partition = -1;
    }

    if (blocks[neighbour.first].fully_loaded_multi_constraints(curr_node_weight)) {
      // Partition already fully loaded so dont pre-assign
      return node_partition = -1;
    }
  }

  if(config.one_pass_algorithm == ONEPASS_FENNEL_MULT_OBJ) {
    (*config.stream_edges_assign)[curr_node_id][node_partition] = true;
    for (auto & neighbours : config.curr_node_neighbours) {
      (*config.stream_edges_assign)[neighbours][node_partition] = true;
    }
  }
  
  return set_decision_partitioning(node_partition, curr_node_id, curr_node_weight, 0);
}

PartitionID vertex_partitioning::pre_assign_edge( LongNodeID curr_node_id, 
                                                  std::vector<NodeWeight> & curr_edge_weight,
                                                  std::vector<std::pair<PartitionID, EdgeWeight>> &neighbours,
                                                  HeiClus::PartitionConfig &config,
                                                  std::vector<PartitionID> & com2part,
                                                  std::vector<PartitionID> & cluster_assignment) {
  
  // Note : In edge partitioning neighbour.first is the target NodeID and not the partition to which it belongs.

  PartitionID curr_cluster = cluster_assignment[curr_node_id];
  PartitionID node_partition = com2part[curr_cluster];

  // save where the prepartitioner assigned the nodes, since it is used by the HDRF obj function of the edge partitioning
  (*config.stream_nodes_assign)[curr_node_id] = node_partition;
  
  for (PartitionID neighbour_ix = 0; neighbour_ix < neighbours.size(); neighbour_ix++) {
    auto & neighbour = neighbours[neighbour_ix];
    
    if(neighbour.first == -1) {
      // finished iterating through all the neighbours
      break;
    }

    PartitionID neighbour_cluster = cluster_assignment[neighbour.first];
    PartitionID neighbour_part = com2part[neighbour_cluster];

    if(node_partition != neighbour_part || neighbour.first < curr_node_id) {
      // edge endpoints dont match or the edge has already been evaluated since curr_node_id > neighbour.first.
      continue;
    }

    curr_edge_weight[0] = 0;
    curr_edge_weight[1] = 0;

    // Adjust the weights associated with assigning edge to node_partition 
    if(!(*config.stream_edges_assign)[curr_node_id][node_partition]) {
      curr_edge_weight[0]++;
    }
    if(!(*config.stream_edges_assign)[neighbour.first][node_partition]) {
      curr_edge_weight[0]++;
    }

    curr_edge_weight[1] = neighbour.second;
    
    if (blocks[node_partition].fully_loaded_edge_multi_constraints(curr_edge_weight)) {
      // Partition already fully loaded so dont pre-assign edge
      return node_partition = -1;
    }

    // Set Edge Partitioning decision.
    node_partition = set_decision_edge_partitioning(node_partition, curr_node_id, curr_edge_weight);

    // Assign the endpoints to the partition.
    (*config.stream_edges_assign)[curr_node_id][node_partition] = true;
    (*config.stream_edges_assign)[neighbour.first][node_partition] = true;
    
    (*config.stream_blocks_weight)[node_partition] += curr_edge_weight[0]; 

    for(int i = 0; i < config.number_of_constraints; i++) {
      (*config.stream_blocks_load)[node_partition][i] += curr_edge_weight[i];
    }   
  }
  
  // all neighbouring endpoints in different partitions, dont pre-assign edge
  return node_partition = -1;
}


