#include "fennel_multi_obj.h"
#include "partition/onepass_partitioning/floating_block.h"
#include "partition/onepass_partitioning/vertex_partitioning.h"
#include "partition/onepass_partitioning/edge_assignment.h"

onepass_fennel_multi_obj::onepass_fennel_multi_obj(PartitionID k0, PartitionID kf,
                               PartitionID max_blocks, NodeID n_threads,
                               bool hashing /*=false*/, float gamma /*=1.5*/)
    : vertex_partitioning(k0, kf, max_blocks, n_threads, hashing) {
  this->gamma = gamma;
}

onepass_fennel_multi_obj::~onepass_fennel_multi_obj() {}

void onepass_fennel_multi_obj::precompute_for_node( LongNodeID curr_node_id, HeiClus::PartitionConfig& config) {
    
    const PartitionID k          = config.k;
    const auto&       neighbours = config.curr_node_neighbours;
    const auto&       sna        = *config.stream_nodes_assign;
    EdgeAssignmentArray& sea     = *config.stream_edges_assign;
    const std::size_t stride     = sea.stride();

    // Size arrays once; thereafter only zero-fill (avoids reallocation every node).
    if (static_cast<PartitionID>(m_neighbor_rep_count.size()) != k) {
        m_neighbor_rep_count.assign(k, 0u);
        m_full_rep_set.assign(k, 0u);
    } else {
        std::fill(m_neighbor_rep_count.begin(), m_neighbor_rep_count.end(), 0u);
        std::fill(m_full_rep_set.begin(), m_full_rep_set.end(), 0u);
    }

    m_degree_count          = static_cast<NodeID>(neighbours.size());
    const uint64_t* curr_row = sea.row_ptr(curr_node_id);

    for (const LongNodeID neighbour : neighbours) {
        // Part 1: row-major scan of this neighbour's bitset to tally
        //         how many neighbours are already replicated in each block.
        //         (was the column-major hot loop: sea[neighbour][my_block_id] × k)
        const uint64_t* n_row = sea.row_ptr(neighbour);
        for (std::size_t w = 0; w < stride; ++w) {
            uint64_t word = n_row[w];
            while (word) {
                int bit = __builtin_ctzll(word);
                m_neighbor_rep_count[w * 64 + bit]++;
                word &= word - 1;   // clear lowest set bit
            }
        }

        // Part 2: does curr_node need to replicate into this neighbour's partition?
        //         (was repeated k times; now done once per neighbour)
        const PartitionID np = sna[neighbour];
        if (np != static_cast<PartitionID>(-1)) {
            if (!((curr_row[np >> 6] >> (np & 63)) & 1u)) {
                m_full_rep_set[np] = 1u;
            }
        }
    }

    // Count total required replications (small k-loop, runs once per node)
    m_pre_rep_count = 0;
    for (PartitionID p = 0; p < k; ++p)
        if (m_full_rep_set[p]) ++m_pre_rep_count;
}

double onepass_fennel_multi_obj::compute_score(
    floating_block& block, int my_thread, HeiClus::PartitionConfig& config,
    PartitionID cluster, int restreaming, LongNodeID curr_node_id,
    std::vector<floating_block>& blocks,
    std::pair<NodeID, EdgeWeight> const& neighbour_node) {
    
    return block.get_fennel_multi_obj_fast(
        m_degree_count,
        config.curr_node_degree,
        m_neighbor_rep_count,
        m_pre_rep_count,
        m_full_rep_set[block.my_block_id],
        config.tau_mult_obj,
        config.k);
}

void onepass_fennel_multi_obj::instantiate_blocks(LongNodeID n, LongEdgeID m,
                                        PartitionID k, PartitionID number_of_constraints, ImbalanceType epsilon, ImbalanceType epsilon_edge, HeiClus::PartitionConfig & config) {
  
  vertex_partitioning::instantiate_blocks(n, m, k, number_of_constraints, epsilon, epsilon_edge, config);
  auto &blocks = vertex_partitioning::blocks;
  for (auto &block : blocks) {
    block.set_fennel_constants(n, m, k, this->gamma);
  }
}
