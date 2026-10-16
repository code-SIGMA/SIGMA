#ifndef HDRFPP__CLUSTRE_HPP_
#define HDRFPP__CLUSTRE_HPP_

#include <fstream>
#include <iostream>
#include <math.h>
#include <regex.h>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <sys/resource.h>
#include <vector>

#include "extern/VieClus/extern/KaHIP/lib/algorithms/cycle_search.h"
#include "extern/VieClus/extern/KaHIP/app/balance_configuration.h"
#include "extern/VieClus/extern/KaHIP/lib/data_structure/graph_access.h"
#include "extern/VieClus/extern/KaHIP/lib/io/graph_io.h"
#include "lib/tools/macros_assertions.h"
#include "extern/VieClus/extern/KaHIP/lib/parallel_mh_clustering/parallel_mh_async_clustering.h"
#include "extern/VieClus/extern/KaHIP/lib/partition/graph_partitioner.h"
#include "extern/VieClus/extern/KaHIP/lib/partition/partition_config.h"
#include "extern/VieClus/extern/KaHIP/lib/tools/quality_metrics.h"
#include "extern/VieClus/extern/KaHIP/lib/tools/random_functions.h"
#include "lib/tools/timer.h"
#include "configuration.h"
 
#include "macros_assertions.h"
#include "data_structure/ExternalPQ.h"
#include "data_structure/graph_access.h"
#include "graph_io_stream.h"
#include "partition/partition_config.h"
#include "quality_metrics.h"
#include "timer.h"
#include "tools/random_functions.h"
#include "graph_io_stream.h"

#include "partition/onepass_partitioning/fennel.h"
#include "partition/onepass_partitioning/modularity.h"
#include "partition/onepass_partitioning/vertex_partitioning.h"
 
#include "extclustering/extclustering.h"
#include "extclustering/extclustering_vieclus.h"
 
//#include "FlatBufferWriter.h"
#include "Stream_CPI_Info_generated.h"
#include "cpi/run_length_compression.hpp"
 
#include "data_structure/compression_vectors/CompressionDataStructure.h"
#include "data_structure/compression_vectors/RunLengthCompressionVector.h"
#include "data_structure/compression_vectors/BatchRunLengthCompression.h"

#include "robin_hood.h"
#include <chrono>

class CluStRE {
private:
    std::vector<PartitionID> communities; // index is vertex id, community of a vertex
    std::vector<NodeWeight> volumes; // index is community id, volume of a community
	std::vector<NodeWeight> external_degrees; // external degree of each community; index is community id
    std::vector<double> quality_scores; // quality of the communities (intra-cluster edges / inter-cluster edges)
    
    PartitionID next_community_id;
    std::ostringstream redirected_cout;

    // Struct to store captured values
    struct CapturedValues {
        std::size_t space_in_bytes;
        std::size_t uncompressed_space_in_bytes;
        double space_in_mib;
        double relative;
    };
    
    HeiClus::PartitionConfig config;

public:
    explicit CluStRE(const HeiClus::PartitionConfig partitionconfig);
    std::vector<PartitionID>* find_communities();
    std::vector<NodeWeight> get_volumes();
    std::vector<double> get_quality_scores();

    void initialize_onepass_partitioner(HeiClus::PartitionConfig &config, vertex_partitioning *&onepass_partitioner);
    void initialize_extclustering(int argn, char **argv, HeiClus::PartitionConfig &config, extclustering *&ext_clusterer);
    std::string extractBaseFilename(const std::string &fullPath);
    CapturedValues parseCapturedValues(const std::string &output_str);
    long getMaxRSS();
    std::ostream &cout_redirect();
    
    void MemoryConsumptionSignificantDS(robin_hood::unordered_flat_map<std::pair<PartitionID, PartitionID>, EdgeWeight, PairHash> quotient,
                                        std::vector<floating_block> &artificial_blocks,
                                        std::vector<NodeWeight> &stream_blocks_weight,
                                        std::shared_ptr<CompressionDataStructure<PartitionID>> &block_assignments,
                                        std::vector<PartitionID> &stream_nodes_assigned,
                                        std::vector<PartitionID> &clusters_mapping,
                                        std::vector<std::vector<std::pair<PartitionID, EdgeWeight>>> &neighbor_blocks,
                                        LongNodeID rle_length);
    
    void RunningTimeSubModules( double &global_mapping_time,
                                double &buffer_io_time,
                                double &total_time,
                                double &ext_alg_time,
                                double &atf_node_construction,
                                double &scr_vec_push_back,
                                double &qgraph_update,
                                double &node_assignments,
                                double &io_label_prop_time);
};


#endif //HDRFPP__CLUSTRE_HPP_
