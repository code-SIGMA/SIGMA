#include "fennel_range.h"
#include "partition/onepass_partitioning/floating_block.h"
#include "partition/onepass_partitioning/vertex_partitioning.h"

onepass_range_fennel::onepass_range_fennel(PartitionID k0, PartitionID kf,
                               PartitionID max_blocks, NodeID n_threads,
                               bool hashing /*=false*/, float gamma /*=1.5*/)
    : vertex_partitioning(k0, kf, max_blocks, n_threads, hashing) {
  this->gamma = gamma;
}

onepass_range_fennel::~onepass_range_fennel() {}

double onepass_range_fennel::compute_score(floating_block &block, int my_thread, HeiClus::PartitionConfig &config, PartitionID cluster, int restreaming, LongNodeID curr_node_id, std::vector<floating_block> & blocks, std::pair<NodeID, EdgeWeight> const& neighbour_node) {
  return block.get_fennel_range_obj(config.curr_node_degree);
}

void onepass_range_fennel::instantiate_blocks(LongNodeID n, LongEdgeID m,
                                        PartitionID k, PartitionID number_of_constraints, ImbalanceType epsilon, ImbalanceType epsilon_edge, HeiClus::PartitionConfig & config) {
  
  vertex_partitioning::instantiate_blocks(n, m, k, number_of_constraints, epsilon, epsilon_edge, config);
  auto &blocks = vertex_partitioning::blocks;
  for (auto &block : blocks) {
    block.set_fennel_constants(n, m, k, this->gamma);
  }
}
