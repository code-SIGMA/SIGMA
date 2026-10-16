#ifndef ONEPASS_RANGE_FENNEL_7I4IR31Y
#define ONEPASS_RANGE_FENNEL_7I4IR31Y

#include <algorithm>

#include "definitions.h"
#include "random_functions.h"
#include "timer.h"

#include "vertex_partitioning.h"

class onepass_range_fennel : public vertex_partitioning {
public:
  onepass_range_fennel(PartitionID k0, PartitionID kf, PartitionID max_blocks,
                 NodeID n_threads, bool hashing = false, float gamma = 1.5);
  virtual ~onepass_range_fennel();
  void instantiate_blocks(LongNodeID n, LongEdgeID m, PartitionID k, PartitionID number_of_constraints,
                          ImbalanceType epsilon, ImbalanceType epsilon_edge, HeiClus::PartitionConfig & config);

protected:
  double compute_score(floating_block &block, int my_thread, HeiClus::PartitionConfig &config, PartitionID cluster, int restreaming, LongNodeID curr_node_id, std::vector<floating_block> & blocks, std::pair<NodeID, EdgeWeight> const& neighbour_node);
  float gamma;
};

#endif /* end of include guard: ONEPASS_RANGE_FENNEL_7I4IR31Y */ 
