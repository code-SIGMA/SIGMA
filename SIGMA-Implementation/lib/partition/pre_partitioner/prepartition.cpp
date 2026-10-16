#include "prepartition.h"

PrePartitionGraph::PrePartitionGraph(HeiClus::PartitionConfig config) {
    prepartition_config = config;

    partition_volume.resize(prepartition_config.k, 0);
}

HeiClus::PartitionConfig PrePartitionGraph::start_prepartitioning() {
    if (prepartition_config.prepartition_graph == CLUSTRE) {
        CluStRE_prepart = new CluStRE(prepartition_config);
        
        // Avoid Deep Copies
        communities = std::move(*(CluStRE_prepart->find_communities()));
        
        // get volumes of the communities
        volumes = CluStRE_prepart->get_volumes(); 
        
        delete CluStRE_prepart;
        CluStRE_prepart = NULL;
    }

    com2part.resize(volumes.size(), -1);

    evaluate_prepartitioner(communities, volumes);

    if (prepartition_config.partitioning_type == NODE_PARTITIONING) {
        sorted_com_node_prepartitioning();
    } else {
        sorted_com_edge_prepartitioning();
    }
    
    return prepartition_config;    
}

void PrePartitionGraph::evaluate_prepartitioner(const std::vector<PartitionID>& coms, const std::vector<NodeWeight>& vols) {

    timer timer_eval;
    timer_eval.restart();
    
    if (&communities != &coms) {
        communities = coms;
    }
    if (&volumes != &vols) {
        volumes = vols;
    }

    // emplace each community with its volume size
    std::vector<std::pair<uint64_t, uint32_t>> sorted_communities;
    for (size_t i = 0; i < volumes.size(); ++i) {
        sorted_communities.emplace_back(volumes[i], i);
    }

    std::sort(sorted_communities.rbegin(), sorted_communities.rend()); // sort in descending order

    for (auto& com_volume_pair : sorted_communities) {
        // the rest of the communities has volume 0
        if (com_volume_pair.first == 0) break; 

        PartitionID min_p = find_min_vol_partition();
        partition_volume[min_p] += com_volume_pair.first;
        
        // assign partition min_p to the current community
        com2part[com_volume_pair.second] = min_p; 
    }
    
    timer_eval.elapsed();
}

void PrePartitionGraph::sorted_com_node_prepartitioning() {
// Stream over the nodes again and just read their neighbourhood. 
// If all the neighbours are in the same assigned cluster as the streamed node then assign streamed node to that cluster.
// If partition load is at capacity or not all neighbours agree then do not pre-assign (parameter can be tweacked to like 70% agreement...)

    std::string graph_filename = prepartition_config.graph_filename;
    std::vector<std::vector<LongNodeID>> *input = NULL;
    EdgeID total_edge_cut = 0;
    EdgeWeight qap = 0;
    double balance = 0;
    double total_nodes = 0;

    int active_nodes_amount = 0;

    timer t, processing_t, io_t, ext_t, atf_t, score_t, qgraph_t, nd_assign_t, io_label_prop_t, ls_frac_time_t, ls_time_t;

    double global_mapping_time = 0;
    double buffer_io_time = 0;
    double total_time = 0;
    double ext_alg_time = 0;
    double atf_node_construction = 0;
    double scr_vec_push_back = 0;
    double qgraph_update = 0;
    double node_assignments = 0;
    double io_lp_time = 0;
    double ls_frac_time = 0;
    double ls_time = 0;

    bool is_graph_weighted = false;
    bool suppress_output = false;
    bool recursive = false;
    bool active_nodes_exist = false;

    // container for storing block assignments used by the streaming algorithm
    std::shared_ptr<CompressionDataStructure<PartitionID>> block_assignments;
    ls_frac_time_t.restart();
    processing_t.restart();

    // Read metadata graph information such as number of nodes, number of edges.
    io_t.restart();
    graph_io_stream::readFirstLineStreamPartitioning(prepartition_config, graph_filename, total_edge_cut, qap);
    buffer_io_time += io_t.elapsed();

    // Initialise the selected score function metric for the streaming clustering algorithm
    vertex_partitioning *onepass_partitioner = new onepass_modularity(0, 0, prepartition_config.total_nodes,
                                                prepartition_config.parallel_nodes, prepartition_config.mode, false, prepartition_config.cpm_gamma);
    
    // set up block assignment container based on algorithm configuration
    if (prepartition_config.rle_length == 0) {
        block_assignments = std::make_shared<RunLengthCompressionVector<PartitionID>>();
    }

    LongNodeID num_lines = 1;
    int restreaming = 0;
    int my_thread = 0;

    onepass_partitioner->instantiate_blocks(prepartition_config.remaining_stream_nodes,
                                            prepartition_config.remaining_stream_edges,
                                            prepartition_config.k,
                                            prepartition_config.number_of_constraints,
                                            prepartition_config.imbalance,
                                            prepartition_config.epsilon_edge,
                                            prepartition_config);

    for (int i = 0; i < prepartition_config.parallel_nodes; i++) {
      prepartition_config.all_blocks_to_keys[i].resize(prepartition_config.k);
      for (auto &b : prepartition_config.all_blocks_to_keys[i]) {
        b = INVALID_PARTITION;
      }
      prepartition_config.neighbor_blocks[i].resize(prepartition_config.k);
      prepartition_config.next_key[i] = 0;
    }

    std::vector<NodeWeight> node_weights(prepartition_config.number_of_constraints, 0);

    for (LongNodeID curr_node = 0; curr_node < prepartition_config.n_batches; curr_node++) {
        std::fill(node_weights.begin(), node_weights.end(), 0);

        // Load 1 line (1 Node) from disk to then serve an the input to the clusterer
        io_t.restart();
        graph_io_stream::loadBufferLinesToBinary(prepartition_config, input, num_lines, curr_node, restreaming);
        buffer_io_time += io_t.elapsed();

        t.restart();

        // Insert artificial nodes for neighbouring nodes that have already been streamed
        atf_t.restart();
        graph_io_stream::readNodeOnePassPreAssignment(prepartition_config, curr_node, my_thread, input, block_assignments, onepass_partitioner, node_weights, com2part, communities);
        atf_node_construction += atf_t.elapsed();

        // Calculate score and assign node to cluster with highest score gain
        score_t.restart();
        PartitionID block = onepass_partitioner->pre_assign_node(curr_node, node_weights, prepartition_config.neighbor_blocks[my_thread], prepartition_config, com2part, communities);
        scr_vec_push_back += score_t.elapsed();

        // Assign the decision cluster to node.
        nd_assign_t.restart();
        if (prepartition_config.rle_length == -1) {
            (*prepartition_config.stream_nodes_assign)[curr_node] = block;
        }
        else if (prepartition_config.rle_length == 0) {
                block_assignments->Append(block);
        }
        node_assignments += nd_assign_t.elapsed();

        // Delete currently streamed node from memory
        if (!prepartition_config.ram_stream) {
            delete input;
        }

        if(block != -1) {
            (*prepartition_config.stream_blocks_weight)[block] += 1;
            
            for(int i = 0; i < prepartition_config.number_of_constraints; i++) {
                (*prepartition_config.stream_blocks_load)[block][i] += node_weights[i];
            }
        }
        global_mapping_time += t.elapsed();
    }
    delete onepass_partitioner;
}

void PrePartitionGraph::sorted_com_edge_prepartitioning() {
// Stream over the nodes again and just read a single edge at a time. 
// assign edges to block 'i' if both its end points are in block 'i'. 
// We assign edges to block i, by assigning their endpoints to block i.

    std::string graph_filename = prepartition_config.graph_filename;
    std::vector<std::vector<LongNodeID>> *input = NULL;
    EdgeID total_edge_cut = 0;
    EdgeWeight qap = 0;
    double balance = 0;
    double total_nodes = 0;

    int active_nodes_amount = 0;

    timer t, processing_t, io_t, ext_t, atf_t, score_t, qgraph_t, nd_assign_t, io_label_prop_t, ls_frac_time_t, ls_time_t;

    double global_mapping_time = 0;
    double buffer_io_time = 0;
    double total_time = 0;
    double ext_alg_time = 0;
    double atf_node_construction = 0;
    double scr_vec_push_back = 0;
    double qgraph_update = 0;
    double node_assignments = 0;
    double io_lp_time = 0;
    double ls_frac_time = 0;
    double ls_time = 0;

    bool is_graph_weighted = false;
    bool suppress_output = false;
    bool recursive = false;
    bool active_nodes_exist = false;

    // container for storing block assignments used by the streaming algorithm
    std::shared_ptr<CompressionDataStructure<PartitionID>> block_assignments;
    ls_frac_time_t.restart();
    processing_t.restart();

    // Read metadata graph information such as number of nodes, number of edges.
    io_t.restart();
    graph_io_stream::readFirstLineStreamPartitioning(prepartition_config, graph_filename, total_edge_cut, qap);
    buffer_io_time += io_t.elapsed();

    // Initialise the selected score function metric for the streaming clustering algorithm
    vertex_partitioning *onepass_partitioner = new onepass_modularity(0, 0, prepartition_config.total_nodes,
                                                prepartition_config.parallel_nodes, prepartition_config.mode, false, prepartition_config.cpm_gamma);
    
    // set up block assignment container based on algorithm configuration
    if (prepartition_config.rle_length == 0) {
        block_assignments = std::make_shared<RunLengthCompressionVector<PartitionID>>();
    }

    LongNodeID num_lines = 1;
    int restreaming = 0;
    int my_thread = 0;

    onepass_partitioner->instantiate_blocks(prepartition_config.remaining_stream_nodes,
                                            prepartition_config.remaining_stream_edges,
                                            prepartition_config.k,
                                            prepartition_config.number_of_constraints,
                                            prepartition_config.imbalance,
                                            prepartition_config.epsilon_edge,
                                            prepartition_config );
    
    for (int i = 0; i < prepartition_config.parallel_nodes; i++) {
        prepartition_config.all_blocks_to_keys[i].resize(prepartition_config.k);
        for (auto &b : prepartition_config.all_blocks_to_keys[i]) {
            b = INVALID_PARTITION;
        }
        prepartition_config.neighbor_blocks[i].resize(prepartition_config.k);
        prepartition_config.next_key[i] = 0;
    }

    std::vector<NodeWeight> edge_weights(prepartition_config.number_of_constraints, 0);

    for (LongNodeID curr_node = 0; curr_node < prepartition_config.n_batches; curr_node++) {
        std::fill(edge_weights.begin(), edge_weights.end(), 0);

        // Load 1 line (1 Node) from disk to then serve an the input to the clusterer
        io_t.restart();
        graph_io_stream::loadBufferLinesToBinary(prepartition_config, input, num_lines, curr_node, restreaming);
        buffer_io_time += io_t.elapsed();

        // Insert artificial nodes for neighbouring nodes that have already been streamed
        atf_t.restart();
        graph_io_stream::readEdgeOnePassPartitioning(prepartition_config, curr_node, my_thread, input, block_assignments, onepass_partitioner, edge_weights);
        atf_node_construction += atf_t.elapsed();

        // Calculate score and assign node to cluster with highest score gain
        score_t.restart();
        PartitionID block;
        block = onepass_partitioner->pre_assign_edge(curr_node, edge_weights, prepartition_config.neighbor_blocks[my_thread], prepartition_config, com2part, communities);
        scr_vec_push_back += score_t.elapsed();

        // Delete currently streamed node from memory
        if (!prepartition_config.ram_stream) {
            delete input;
        }
    }
    delete onepass_partitioner;
}


PartitionID PrePartitionGraph::find_min_vol_partition() {
    
    auto min_p_volume = partition_volume[0];
    int min_p = 0;
    for (int i = 1; i < prepartition_config.k; i++) {
        if (partition_volume[i] < min_p_volume) {
            min_p_volume = partition_volume[i];
            min_p = i;
        }
    }
    return min_p;    
}