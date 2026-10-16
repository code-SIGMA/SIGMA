#ifndef ONEPASS_FENNEL_MULTI_OBJ_7I4IR31Y
#define ONEPASS_FENNEL_MULTI_OBJ_7I4IR31Y

#include <algorithm>
#include <vector>

#include "definitions.h"
#include "random_functions.h"
#include "timer.h"

#include "vertex_partitioning.h"

class onepass_fennel_multi_obj : public vertex_partitioning {
public:
  onepass_fennel_multi_obj(PartitionID k0, PartitionID kf, PartitionID max_blocks,
                 NodeID n_threads, bool hashing = false, float gamma = 1.5);
  virtual ~onepass_fennel_multi_obj();
  void instantiate_blocks(LongNodeID n, LongEdgeID m, PartitionID k, PartitionID number_of_constraints,
                          ImbalanceType epsilon, ImbalanceType epsilon_edge, HeiClus::PartitionConfig & config);

protected:
  void precompute_for_node(LongNodeID curr_node_id,
                           HeiClus::PartitionConfig& config) override;
  double compute_score(floating_block &block, int my_thread, HeiClus::PartitionConfig &config, PartitionID cluster, int restreaming, LongNodeID curr_node_id, std::vector<floating_block> & blocks, std::pair<NodeID, EdgeWeight> const& neighbour_node) override;
  float gamma;

private:
  // Per-node precomputed data (filled by precompute_for_node, consumed by compute_score)
  std::vector<NodeID>   m_neighbor_rep_count;  // [k]: # neighbours already replicated in block p
  std::vector<uint8_t>  m_full_rep_set;        // [k]: 1 if curr_node must replicate into block p
  NodeID                m_pre_rep_count = 0;   // popcount of m_full_rep_set
  NodeID                m_degree_count  = 0;   // # entries in curr_node_neighbours (integer count)
};

#endif /* end of include guard: ONEPASS_RANGE_FENNEL_7I4IR31Y */
