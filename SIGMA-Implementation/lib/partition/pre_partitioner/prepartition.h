#ifndef PREPARTITION_7I4IR31Y
#define PREPARTITION_7I4IR31Y

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

#include "partition/pre_partitioner/clustre_prepartition.h"

class PrePartitionGraph{
public:
    PrePartitionGraph(HeiClus::PartitionConfig config);

    // Here I just call basically the prepartitioner that was chosen and apply the find communities function
    HeiClus::PartitionConfig start_prepartitioning();

    // Here after the communities are known we then preassign some nodes (look exactly how this is done)
    void evaluate_prepartitioner(const std::vector<PartitionID>& coms, const std::vector<NodeWeight>& vols);

    // One more stream to go over all the edges and place them.
    void sorted_com_node_prepartitioning();
    void sorted_com_edge_prepartitioning();

    PartitionID find_min_vol_partition();


private:
    HeiClus::PartitionConfig prepartition_config;
    CluStRE * CluStRE_prepart = NULL; 
    std::vector<PartitionID> communities;
    std::vector<NodeWeight> volumes;

    std::vector<NodeWeight> partition_volume;
    std::vector<PartitionID> com2part;

    // edge_load.resize(globals.NUM_PARTITIONS, 0);
    // vertex_partition_matrix.resize(globals.NUM_VERTICES);

};

#endif