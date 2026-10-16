/******************************************************************************
 * graph_io_stream.cpp
 * *
 * Source of KaHIP -- Karlsruhe High Quality Partitioning.
 * Marcelo Fonseca Faraj
 *****************************************************************************/

#include "graph_io_stream.h"
#include "cpi/run_length_compression.hpp"
#include "timer.h"
#include <math.h>
#include <sstream>
 
//#include <kagen.h>
//#include <mpi.h>

#define MIN(A, B) (((A) < (B)) ? (A) : (B))
#define MAX(A, B) (((A) > (B)) ? (A) : (B))

graph_io_stream::graph_io_stream() {}

graph_io_stream::~graph_io_stream() {}

bool graph_io_stream::hasEnding(std::string const &string, std::string const &ending) {
    if (string.length() >= ending.length()) {
        return (0 == string.compare(string.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

void graph_io_stream::restreamingFileReset(HeiClus::PartitionConfig &partition_config,
                                          std::string graph_filename ) {
  if (partition_config.stream_in != NULL) {
    delete partition_config.stream_in;
  }

  std::string bin_ending(".bin");
  std::string parhip_ending(".parhip");
  if (hasEnding(graph_filename, bin_ending) || hasEnding(graph_filename, parhip_ending)) {
      std::vector<unsigned long long> buffer(3, 0);
      partition_config.stream_in = new std::ifstream(graph_filename.c_str(), std::ios::binary | std::ios::in);;
      if ((*partition_config.stream_in)) {
          (*partition_config.stream_in).read((char *) (&buffer[0]), 3 * sizeof(unsigned long long));
      }

      partition_config.bin_start_pos = 3 * sizeof(unsigned long long);

  } else {
    partition_config.stream_in = new std::ifstream(graph_filename.c_str());
    if (!(*(partition_config.stream_in))) {
      std::cerr << "Error opening " << graph_filename << std::endl;
      exit(1);
    }
    std::vector<std::string> * lines;

    lines = new std::vector<std::string>(1);
    std::getline(*(partition_config.stream_in), (*lines)[0]);

    // skip comments
    while ((*lines)[0][0] == '%') {
      std::getline(*(partition_config.stream_in), (*lines)[0]);
    }

    if (partition_config.partialOffsets == NULL &&
       (partition_config.mode == LIGHT_PLUS || partition_config.mode == STRONG)) {
      partition_config.partialOffsets = new std::vector<std::streampos>();
      partition_config.partialOffsets->reserve((partition_config.total_nodes / partition_config.offset_interval) + 1);
    }
    
    delete lines;
  }

  if( partition_config.activeNodes_set == NULL &&
      (partition_config.mode == LIGHT_PLUS || partition_config.mode == STRONG)) {
    partition_config.activeNodes_set = new robin_hood::unordered_set<LongNodeID>();
  }

}

void graph_io_stream::readFirstLineStreamClustering(HeiClus::PartitionConfig &partition_config,
                                          std::string graph_filename,
                                          EdgeID &total_edge_cut,
                                          EdgeWeight &qap) {
  
  if (partition_config.stream_in != NULL) {
      delete partition_config.stream_in;
  }

  std::string bin_ending(".bin");
  std::string parhip_ending(".parhip");
  if (hasEnding(graph_filename, bin_ending) || hasEnding(graph_filename, parhip_ending)) {
      std::vector<unsigned long long> buffer(3, 0);
      partition_config.stream_in = new std::ifstream(graph_filename.c_str(), std::ios::binary | std::ios::in);;
      if ((*partition_config.stream_in)) {
          (*partition_config.stream_in).read((char *) (&buffer[0]), 3 * sizeof(unsigned long long));
      }

      unsigned long long version = buffer[0];
      partition_config.remaining_stream_nodes = static_cast<NodeID>(buffer[1]);
      partition_config.remaining_stream_edges = static_cast<NodeID>(buffer[2]) / 2;

      partition_config.bin_start_pos = 3 * sizeof(unsigned long long);
  } else {
    partition_config.stream_in = new std::ifstream(graph_filename.c_str());
    if (!(*(partition_config.stream_in))) {
      std::cerr << "Error opening " << graph_filename << std::endl;
      exit(1);
    }
    std::vector<std::string> *lines;

    lines = new std::vector<std::string>(1);
    std::getline(*(partition_config.stream_in), (*lines)[0]);

    // skip comments
    while ((*lines)[0][0] == '%') {
      std::getline(*(partition_config.stream_in), (*lines)[0]);
    }

    std::stringstream ss((*lines)[0]);
    ss >> partition_config.remaining_stream_nodes;
    ss >> partition_config.remaining_stream_edges;
    ss >> partition_config.remaining_stream_ew;

    delete lines;
  }

  switch (partition_config.remaining_stream_ew) {
  case 1:
    partition_config.read_ew = true;
    break;
  case 10:
    partition_config.read_nw = true;
    break;
  case 11:
    partition_config.read_ew = true;
    partition_config.read_nw = true;
    break;
  }

  partition_config.total_edges = partition_config.remaining_stream_edges;
  partition_config.total_nodes = partition_config.remaining_stream_nodes;

  if(partition_config.max_num_clusters == -1) {
    partition_config.max_num_clusters = partition_config.total_nodes * partition_config.cluster_fraction;
  }

  // Index value INVALID_PARTITION => node has not been streamed yet.
  if (partition_config.stream_nodes_assign == NULL && partition_config.rle_length == -1) {
       partition_config.stream_nodes_assign = new std::vector<PartitionID>(
               partition_config.remaining_stream_nodes, INVALID_PARTITION);
  }

  if (partition_config.stream_blocks_weight == NULL) {
    partition_config.stream_blocks_weight =
        new std::vector<NodeWeight>(0, 0);
  }

  // when using CluStRE consider both node and edge upperbounds.
  partition_config.max_cluster_node_volume = ( (100 + partition_config.epsilon) / 100 ) * ( partition_config.total_nodes / partition_config.k ) + 1;
  partition_config.max_cluster_edge_volume = ( (100 + partition_config.epsilon_edge) / 100 ) * ( ( partition_config.total_edges * 2) / partition_config.k ) + 1;

  partition_config.total_stream_nodeweight = 0;
  partition_config.total_stream_nodecounter = 0;
  partition_config.stream_n_nodes = partition_config.remaining_stream_nodes;

  if (partition_config.num_streams_passes >
      1 + partition_config.restream_number) {
    partition_config.stream_total_upperbound = ceil(
        ((100 + 1.5 * partition_config.imbalance) / 100.) *
        (partition_config.remaining_stream_nodes / (double)partition_config.k));
  } else {
    partition_config.stream_total_upperbound = ceil(
        ((100 + partition_config.imbalance) / 100.) *
        (partition_config.remaining_stream_nodes / (double)partition_config.k));
  }

  partition_config.fennel_alpha =
      partition_config.remaining_stream_edges *
      std::pow(partition_config.k, partition_config.fennel_gamma - 1) /
      (std::pow(partition_config.remaining_stream_nodes,
                partition_config.fennel_gamma));

  partition_config.fennel_alpha_gamma =
      partition_config.fennel_alpha * partition_config.fennel_gamma;

  if (partition_config.full_stream_mode && !partition_config.restream_number) {
    partition_config.quotient_nodes = 0;
  } else {
    partition_config.quotient_nodes = partition_config.k;
  }

  total_edge_cut = 0;
  qap = 0;
  if (partition_config.stream_buffer_len ==
      0) { // signal of partial restream standard buffer size
    partition_config.stream_buffer_len = (LongNodeID)ceil(
        partition_config.remaining_stream_nodes / (double)partition_config.k);
  }
  partition_config.nmbNodes = MIN(partition_config.stream_buffer_len,
                                  partition_config.remaining_stream_nodes);
  partition_config.n_batches = ceil(partition_config.remaining_stream_nodes /
                                    (double)partition_config.nmbNodes);
  partition_config.curr_batch = 0;
}

void graph_io_stream::readFirstLineStreamPartitioning(HeiClus::PartitionConfig &partition_config,
                                          std::string graph_filename,
                                          EdgeID &total_edge_cut,
                                          EdgeWeight &qap) {

  if (partition_config.stream_in != NULL) {
    delete partition_config.stream_in;
  }
  partition_config.stream_in = new std::ifstream(graph_filename.c_str());
  if (!(*(partition_config.stream_in))) {
    std::cerr << "Error opening " << graph_filename << std::endl;
    exit(1);
  }
  std::vector<std::string> *lines;

  lines = new std::vector<std::string>(1);
  std::getline(*(partition_config.stream_in), (*lines)[0]);

  // skip comments
  while ((*lines)[0][0] == '%') {
    std::getline(*(partition_config.stream_in), (*lines)[0]);
  }

  std::stringstream ss((*lines)[0]);
  ss >> partition_config.remaining_stream_nodes;
  ss >> partition_config.remaining_stream_edges;
  ss >> partition_config.remaining_stream_ew;

  switch (partition_config.remaining_stream_ew) {
  case 1:
    partition_config.read_ew = true;
    break;
  case 10:
    partition_config.read_nw = true;
    break;
  case 11:
    partition_config.read_ew = true;
    partition_config.read_nw = true;
    break;
  }

  partition_config.total_edges = partition_config.remaining_stream_edges;
  partition_config.total_nodes = partition_config.remaining_stream_nodes;

  if (partition_config.stream_nodes_assign == NULL && partition_config.rle_length == -1 && partition_config.partitioning_type == NODE_PARTITIONING) {
    partition_config.stream_nodes_assign = new std::vector<PartitionID>(partition_config.remaining_stream_nodes, INVALID_PARTITION);
  } else if (partition_config.stream_edges_assign == NULL && partition_config.rle_length == -1 && partition_config.partitioning_type == EDGE_PARTITIONING) {
    partition_config.stream_edges_assign = new EdgeAssignmentArray(partition_config.remaining_stream_nodes, partition_config.k);
    partition_config.stream_nodes_assign = new std::vector<PartitionID>(partition_config.remaining_stream_nodes, INVALID_PARTITION);
  }

  if(partition_config.stream_edges_assign == NULL && partition_config.rle_length == -1 && partition_config.one_pass_algorithm == ONEPASS_FENNEL_MULT_OBJ) {
    partition_config.stream_edges_assign = new EdgeAssignmentArray(partition_config.remaining_stream_nodes, partition_config.k);
  }
    
  if (partition_config.stream_blocks_weight == NULL) {
    partition_config.stream_blocks_weight = new std::vector<NodeWeight>(partition_config.k, 0);
  }

  if (partition_config.stream_blocks_load == NULL) {
    partition_config.stream_blocks_load = new std::vector<std::vector<NodeWeight>>(partition_config.k,
        std::vector<NodeWeight>(partition_config.number_of_constraints, 0)
    );  
  }

  if(partition_config.with_aw && partition_config.stream_nodes_aw == NULL) {
      partition_config.stream_nodes_aw = new std::vector<bool>(
        partition_config.total_nodes, false);
  }

  if(partition_config.partitioning_type == EDGE_PARTITIONING && partition_config.node_degrees == NULL) {
    partition_config.node_degrees = new std::vector<EdgeID>(partition_config.remaining_stream_nodes, INVALID_PARTITION);
  }

  partition_config.total_stream_nodeweight = 0;
  partition_config.total_stream_nodecounter = 0;
  partition_config.stream_n_nodes = partition_config.remaining_stream_nodes;

  if (partition_config.num_streams_passes >
      1 + partition_config.restream_number) {
    partition_config.stream_total_upperbound = ceil(
        ((100 + 1.5 * partition_config.imbalance) / 100.) *
        (partition_config.remaining_stream_nodes / (double)partition_config.k));
  } else {
    partition_config.stream_total_upperbound = ceil(
        ((100 + partition_config.imbalance) / 100.) *
        (partition_config.remaining_stream_nodes / (double)partition_config.k));
  }

  partition_config.fennel_alpha =
      partition_config.remaining_stream_edges *
      std::pow(partition_config.k, partition_config.fennel_gamma - 1) /
      (std::pow(partition_config.remaining_stream_nodes,
                partition_config.fennel_gamma));

  partition_config.fennel_alpha_gamma =
      partition_config.fennel_alpha * partition_config.fennel_gamma;

  if (partition_config.full_stream_mode && !partition_config.restream_number) {
    partition_config.quotient_nodes = 0;
  } else {
    partition_config.quotient_nodes = partition_config.k;
  }

  total_edge_cut = 0;
  qap = 0;
  if (partition_config.stream_buffer_len ==
      0) { // signal of partial restream standard buffer size
    partition_config.stream_buffer_len = (LongNodeID)ceil(
        partition_config.remaining_stream_nodes / (double)partition_config.k);
  }
  partition_config.nmbNodes = MIN(partition_config.stream_buffer_len,
                                  partition_config.remaining_stream_nodes);
  partition_config.n_batches = ceil(partition_config.remaining_stream_nodes /
                                    (double)partition_config.nmbNodes);
  partition_config.curr_batch = 0;

  delete lines;
}

void graph_io_stream::streamEvaluateClustering(HeiClus::PartitionConfig &config,
                                              const std::string &filename,
                                              vertex_partitioning *onepass_partitioner,
                                              const std::shared_ptr<CompressionDataStructure<PartitionID>>& block_assignments) {
  
  LongNodeID nmbNodes;
  LongEdgeID nmbEdges;
  int ew = 0;

  std::vector<NodeID>blocks_weights(0);
  std::vector<std::pair<EdgeWeight, EdgeWeight>>blocks_sigma(0); //blocks_sigma[i].first -> small Sigma, blocks_sigma[i].second -> big Sigma 
  NodeWeight total_nodeweight = 0;
  //EdgeWeight total_edgeweight = 0;
  uint64_t total_edgeweight = 0;

  std::string bin_ending(".bin");
  std::string parhip_ending(".parhip");
  std::ifstream *in;

  // Check if input file is a binary
  if (hasEnding(config.graph_filename, bin_ending) || hasEnding(config.graph_filename, parhip_ending)) {
    std::vector<unsigned long long> buffer(3, 0);
    in = new std::ifstream(config.graph_filename.c_str(), std::ios::binary | std::ios::in);;
    if ((*in)) {
      (*in).read((char *) (&buffer[0]), 3 * sizeof(unsigned long long));
    }
    
    unsigned long long version = buffer[0];
    nmbNodes = static_cast<NodeID>(buffer[1]);
    nmbEdges = static_cast<NodeID>(buffer[2]) / 2;

    NodeID remaining_nodes = nmbNodes;
    NodeID batch_size = 1;
    NodeID node = 0;
    NodeID target;
    unsigned long long start_pos = 3 * sizeof(unsigned long long);

    while (node < remaining_nodes) {
      unsigned long long nodes_in_batch = static_cast<unsigned long long>(batch_size);
      unsigned long long *vertex_offsets = new unsigned long long[nodes_in_batch + 1];
      (*in).seekg(start_pos);
      (*in).read((char *) (vertex_offsets), (nodes_in_batch + 1) * sizeof(unsigned long long));
      unsigned long long next_pos = start_pos + (nodes_in_batch) * sizeof(unsigned long long);

      unsigned long long edge_start_pos = vertex_offsets[0];
      unsigned long long num_reads = vertex_offsets[nodes_in_batch] - vertex_offsets[0];
      unsigned long long num_edges_to_read = num_reads / sizeof(unsigned long long);
      unsigned long long *edges = new unsigned long long[num_edges_to_read]; // we also need the next vertex offset
      (*in).seekg(edge_start_pos);
      (*in).read((char *) (edges), (num_edges_to_read) * sizeof(unsigned long long));
      start_pos = next_pos;

      unsigned long long pos = 0;

      PartitionID partitionIDSource;
      if(config.rle_length == -1) {
        partitionIDSource = (*config.stream_nodes_assign)[node];
      } else if (config.rle_length == 0) {
        partitionIDSource = block_assignments->GetValueByIndex(node);
      }

      while(blocks_weights.size() <= partitionIDSource) {
        blocks_weights.emplace_back(0);
        blocks_sigma.emplace_back(std::make_pair(0, 0));
      }

      NodeWeight weight = 1;

      unsigned long long degree = (vertex_offsets[1] - vertex_offsets[0]) / sizeof(unsigned long long);
      for (unsigned long long j = 0; j < degree; j++, pos++) {
        auto target = static_cast<NodeID>(edges[pos]);
          
        EdgeWeight edge_weight = 1;
        total_edgeweight += edge_weight;
          
        PartitionID partitionIDTarget;
        if(config.rle_length == -1) {
          partitionIDTarget = (*config.stream_nodes_assign)[target];
        } else if (config.rle_length == 0) {
          partitionIDTarget = block_assignments->GetValueByIndex(target);
        }

        if (partitionIDSource == partitionIDTarget) {
          blocks_sigma[partitionIDSource].first++;
        }
        blocks_sigma[partitionIDSource].second++;
      }
      delete vertex_offsets;
      delete edges;
      node++;
    }

  } else {
 
    std::vector<std::vector<LongNodeID>> *input;
    std::vector<std::string> *lines;
    lines = new std::vector<std::string>(1);
    LongNodeID node_counter = 0;
    buffered_input *ss2 = NULL;
    std::string line;
    std::ifstream in(filename.c_str());

    if (!in) {
      std::cerr << "Error opening " << filename << std::endl;
      return 1;
    }
  
    std::getline(in, (*lines)[0]);
    while ((*lines)[0][0] == '%') {
      std::getline(in, (*lines)[0]); // a comment in the file
    }
  
    std::stringstream ss((*lines)[0]);
    ss >> nmbNodes;
    ss >> nmbEdges;
    ss >> ew;

    bool read_ew = false;
    bool read_nw = false;
    if (ew == 1) {
      read_ew = true;
    } else if (ew == 11) {
      read_ew = true;
      read_nw = true;
    } else if (ew == 10) {
      read_nw = true;
    }
    
    LongNodeID target;

    while (std::getline(in, (*lines)[0])) {
      if ((*lines)[0][0] == '%') {
        continue; // a comment in the file
      }
      LongNodeID node = node_counter++;

      PartitionID partitionIDSource;
      if(config.rle_length == -1) {
        partitionIDSource = (*config.stream_nodes_assign)[node];
      } else if (config.rle_length == 0) {
        partitionIDSource = block_assignments->GetValueByIndex(node);
      }
      
      input = new std::vector<std::vector<LongNodeID>>(1);
      ss2 = new buffered_input(lines);
      ss2->simple_scan_line((*input)[0]);
      std::vector<LongNodeID> &line_numbers = (*input)[0];
      LongNodeID col_counter = 0;

      while(blocks_weights.size() <= partitionIDSource) {
        blocks_weights.emplace_back(0);
        blocks_sigma.emplace_back(std::make_pair(0, 0));
      }
      blocks_weights[partitionIDSource]++;

      NodeWeight weight = 1;
      if (read_nw) {
        weight = line_numbers[col_counter++];
        total_nodeweight += weight;
      }
      while (col_counter < line_numbers.size()) {
        target = line_numbers[col_counter++];
        target = target - 1;
        EdgeWeight edge_weight = 1;
        
        if (read_ew) {
          edge_weight = line_numbers[col_counter++];
        }
        
        total_edgeweight += edge_weight;
        PartitionID partitionIDTarget;
        if(config.rle_length == -1) {
          partitionIDTarget = (*config.stream_nodes_assign)[target];
        } else if (config.rle_length == 0) {
          partitionIDTarget = block_assignments->GetValueByIndex(target);
        }

        if (partitionIDSource == partitionIDTarget) {
          blocks_sigma[partitionIDSource].first++;
        }
        blocks_sigma[partitionIDSource].second++;
      }

      (*lines)[0].clear();
      delete ss2;
      delete input;
      if (in.eof()) {
        break;
      }
    }
    delete lines;
  }

  config.score = onepass_partitioner->calculate_overall_score(blocks_weights, blocks_sigma, nmbEdges);
}

void graph_io_stream::writeClusteringStream(HeiClus::PartitionConfig &config,
                                           const std::string &filename,
                                           vertex_partitioning *onepass_partitioner,
                                           const std::shared_ptr<CompressionDataStructure<PartitionID>>& block_assignments ) {
  std::ofstream f(filename.c_str());

  if(config.rle_length == -1) {
    for (int node = 0; node < config.total_nodes; node++) {
        f << (*config.stream_nodes_assign)[node] << "\n";
    }
  } else if(config.rle_length == 0) {
    for (int node = 0; node < config.total_nodes; node++) {
        f << block_assignments->GetValueByIndex(node) << "\n";
    }
  }
  
  // robin_hood::unordered_map<std::pair<PartitionID, PartitionID>, EdgeWeight>
  /*PartitionID source = 0;
  for(PartitionID s_cluster = 0; s_cluster < onepass_partitioner->blocks.size(); s_cluster++) {
      f <<"Cluster "<<s_cluster<<" neighbours : ";
      for(PartitionID t_cluster = 0; t_cluster < onepass_partitioner->blocks.size(); t_cluster++) {
        if(s_cluster > t_cluster) {continue;}
        std::pair<PartitionID, PartitionID> key = std::make_pair(s_cluster, t_cluster);
        f << ", Connection to " << t_cluster << " with weight " << (*onepass_partitioner->quotient)[key];
    }
    f << std::endl;
  }*/

  f.close();

}

std::shared_ptr<CompressionDataStructure<PartitionID>> graph_io_stream::readClustering(HeiClus::PartitionConfig &config,
                                    const std::string &filename) {
  
  std::string line;
  (*config.stream_blocks_weight).clear();
  std::shared_ptr<CompressionDataStructure<PartitionID>> block_assignments = std::make_shared<RunLengthCompressionVector<PartitionID>>();

  // open file for reading
  std::ifstream in(filename.c_str());
  if (!in) {
    std::cerr << "Error opening file" << filename << std::endl;
  }

  for (NodeID node = 0; node < config.total_nodes; node++) {
    // fetch current line
    std::getline(in, line);
    while (line[0] == '%') { // Comments
      std::getline(in, line);
    }
    PartitionID partition = (PartitionID)atol(line.c_str());
    block_assignments->Append(partition);
    while((*config.stream_blocks_weight).size() <= partition) {
      (*config.stream_blocks_weight).emplace_back(0);  
    }
    (*config.stream_blocks_weight)[partition] += 1;
  }

  in.close();
  return block_assignments;
}

void graph_io_stream::readNodeOnePassPartitioning(HeiClus::PartitionConfig &config, LongNodeID curr_node,
                                                  int my_thread,
                                                  std::vector<std::vector<LongNodeID>> *&input,
                                                  const std::shared_ptr<CompressionDataStructure<PartitionID>>& block_assignments,
                                                  vertex_partitioning *onepass_partitioner,
                                                  std::vector<NodeWeight> & node_weights) {
  auto &read_ew = config.read_ew;
  auto &read_nw = config.read_nw;
  LongNodeID target;

  LongNodeID cursor = (config.ram_stream) ? curr_node : 0;

  auto &all_blocks_to_keys = config.all_blocks_to_keys[my_thread];
  auto &next_key = config.next_key[my_thread];
  auto &neighbor_blocks = config.neighbor_blocks[my_thread];

  config.curr_node_neighbours.clear();

  onepass_partitioner->clear_edgeweight_blocks(neighbor_blocks, next_key,
                                               my_thread);
  next_key = 0;

  std::vector<LongNodeID> &line_numbers = (*input)[cursor];
  LongNodeID col_counter = 0;
  node_weights[0] = (read_nw) ? line_numbers[col_counter++] : 1;
  /* total_nodeweight += weight; */

  PartitionID selecting_factor = (1 + (PartitionID)read_ew);
  config.edges = (line_numbers.size() - col_counter) / selecting_factor;
  
  config.curr_node_neighbours.reserve(config.edges);

  float scaling_factor = 1;
  LongNodeID deg = 0;
  while (col_counter < line_numbers.size()) {
    target = line_numbers[col_counter++];

    config.curr_node_neighbours.push_back(target - 1);
    
    deg++;
    EdgeWeight edge_weight = (read_ew) ? line_numbers[col_counter++] : 1.0;
    
    if(config.with_aw && (*config.stream_nodes_aw)[target - 1]) {
      edge_weight = static_cast<EdgeWeight>(edge_weight / 3.0);
    }
    
    PartitionID targetGlobalPar, targetGlobalParTest;
    if (config.rle_length == -1) {
      targetGlobalPar = (*config.stream_nodes_assign)[target - 1];
    } else if (config.rle_length == 0) {
      if ((target - 1) < curr_node) {
          targetGlobalPar = block_assignments->GetValueByIndex(target-1);
      } else {
        targetGlobalPar = INVALID_PARTITION;
      }
    } 
     
    if (targetGlobalPar != INVALID_PARTITION) {
      PartitionID key = all_blocks_to_keys[targetGlobalPar];
      if (key >= next_key || neighbor_blocks[key].first != targetGlobalPar) {
        all_blocks_to_keys[targetGlobalPar] = next_key;
        auto &new_element = neighbor_blocks[next_key];
        new_element.first = targetGlobalPar;
        new_element.second = edge_weight;
        next_key++;
      } else {
        neighbor_blocks[key].second += edge_weight;
      }
    }
  }

  node_weights[1] = deg + 1;
  config.curr_node_degree = deg;

  config.edges_streamed += deg + 1;

  if (deg > config.max_degree)
    config.max_degree = deg;

  for (PartitionID key = 0; key < next_key; key++) {
    auto &element = neighbor_blocks[key];
    onepass_partitioner->load_edge(element.first,
                                   element.second * scaling_factor, my_thread);
  }

  config.remaining_stream_nodes--;
}

void graph_io_stream::writePartitionStream(HeiClus::PartitionConfig &config,
                                           const std::string &filename,
                                           const std::shared_ptr<CompressionDataStructure<PartitionID>>& block_assignments) {
      
      if(config.partitioning_type == NODE_PARTITIONING) {
        writeNodePartitionStream(config, filename, block_assignments);
      } else {
        // Edges are printed to file while streaming
      }
}

void graph_io_stream::writeNodePartitionStream( HeiClus::PartitionConfig &config,
                                                const std::string &filename,
                                                const std::shared_ptr<CompressionDataStructure<PartitionID>>& block_assignments) {
  std::ofstream f(filename.c_str());
  std::cout << "writing partition to " << filename << " ... " << std::endl;

  for (int node = 0; node < config.total_nodes; node++) {
    if (config.rle_length == -1 ||
        (config.evaluate && config.rle_length == -2)) {
      f << (*config.stream_nodes_assign)[node] << "\n";
    } else if (config.rle_length == 0) {
      f << block_assignments->GetValueByIndex(node) << "\n";
    } else {
      f << block_assignments->GetValueByBatchIndex(node / config.rle_length, (node % config.rle_length))
        << "\n";
    }
  }

  f.close();
}

void graph_io_stream::streamEvaluatePartition(HeiClus::PartitionConfig &config,
                                              const std::string &filename,
                                              EdgeID &edgeCut,
                                              EdgeWeight &qap,
                                              const std::shared_ptr<CompressionDataStructure<PartitionID>>& block_assignments,
                                              std::vector<ImbalanceType> & balance,
                                              double & replication_factor) {
  
  if(config.partitioning_type == EDGE_PARTITIONING) {
    NodeID num_total_replicas = 0;  
    for (const auto& bitset : (*config.stream_edges_assign)) {
      num_total_replicas += static_cast<uint64_t>(bitset.count());
    }

    std::cout<<"num of replicas : "<< num_total_replicas - config.total_nodes<<std::endl;

    replication_factor = double(num_total_replicas)/double(config.total_nodes);

    std::vector<ImbalanceType> constraints_bound(config.number_of_constraints, 0);
    constraints_bound[0] = ceil(((100 + config.epsilon) / 100.) * (num_total_replicas / (double)config.k));
    constraints_bound[1] = ceil(((100 + config.epsilon_edge) / 100.) * (config.total_edges / (double)config.k));
  
    std::vector<ImbalanceType> max(config.number_of_constraints, -1);
    std::vector<ImbalanceType> total_weight(config.number_of_constraints, 0);

    for (int i = 0; i < config.k; i++) {
        for(int j = 0; j < config.number_of_constraints; j++) {
          if ((*config.stream_blocks_load)[i][j] > max[j]) {
              max[j] = (*config.stream_blocks_load)[i][j];
          }
          total_weight[j] += (*config.stream_blocks_load)[i][j];
        }
    }

    std::vector<ImbalanceType> perfect_balance(config.number_of_constraints, 0);
    perfect_balance[0] = ceil((num_total_replicas / (double)config.k));
    perfect_balance[1] = ceil((config.total_edges / (double)config.k));
  
    for(int i = 0; i < config.number_of_constraints; i++) {
      balance[i] = max[i] / perfect_balance[i];
    }

  } else {
    
    EdgeAssignmentArray * stream_edges_assign = nullptr;
    if(config.one_pass_algorithm == ONEPASS_FENNEL_MULT_OBJ) {
      stream_edges_assign = config.stream_edges_assign;
      std::for_each(stream_edges_assign->begin(), stream_edges_assign->end(),[](auto bs){ bs.reset(); });
    } else {
      stream_edges_assign = new EdgeAssignmentArray(config.total_nodes, config.k);
    }

    std::vector<std::vector<LongNodeID>> *input;
    std::vector<std::string> *lines;
    lines = new std::vector<std::string>(1);
    LongNodeID node_counter = 0;
    buffered_input *ss2 = NULL;
    std::string line;
    std::ifstream in(filename.c_str());
    if (!in) {
      std::cerr << "Error opening " << filename << std::endl;
      return 1;
    }
    long nmbNodes;
    long nmbEdges;
    int ew = 0;
    std::getline(in, (*lines)[0]);
    while ((*lines)[0][0] == '%')
      std::getline(in, (*lines)[0]); // a comment in the file

    std::stringstream ss((*lines)[0]);
    ss >> nmbNodes;
    ss >> nmbEdges;
    ss >> ew;
    bool read_ew = false;
    bool read_nw = false;
    if (ew == 1) {
      read_ew = true;
    } else if (ew == 11) {
      read_ew = true;
      read_nw = true;
    } else if (ew == 10) {
      read_nw = true;
    }
    NodeID target;
    NodeWeight total_nodeweight = 0;
    EdgeWeight total_edgeweight = 0;
    edgeCut = 0;
    qap = 0;
    
    std::vector<std::vector<NodeID>> block_weights(config.k,
        std::vector<NodeID>(config.number_of_constraints)
    );

    while (std::getline(in, (*lines)[0])) {
      if ((*lines)[0][0] == '%')
        continue; // a comment in the file
      NodeID node = node_counter++;

      PartitionID partitionIDSource;

      if (config.rle_length == -1 ||
          (config.evaluate && config.rle_length == -2)) {
        partitionIDSource = (*config.stream_nodes_assign)[node];
      } else if (config.rle_length == 0) {
        partitionIDSource = block_assignments->GetValueByIndex(node);
      } else {
          partitionIDSource = block_assignments->GetValueByBatchIndex(node / config.rle_length,
                                                                      (node % config.rle_length));
      }
      input = new std::vector<std::vector<LongNodeID>>(1);
      ss2 = new buffered_input(lines);
      ss2->simple_scan_line((*input)[0]);
      std::vector<LongNodeID> &line_numbers = (*input)[0];
      LongNodeID col_counter = 0;

      (*stream_edges_assign)[node][partitionIDSource] = true;
      
      block_weights[partitionIDSource][0]++;
      block_weights[partitionIDSource][1] += line_numbers.size() + 1;

      NodeWeight weight = 1;
      if (read_nw) {
        weight = line_numbers[col_counter++];
        total_nodeweight += weight;
      }
      while (col_counter < line_numbers.size()) {
        target = line_numbers[col_counter++];
        target = target - 1;
        EdgeWeight edge_weight = 1;
        if (read_ew) {
          edge_weight = line_numbers[col_counter++];
        }
        total_edgeweight += edge_weight;
        PartitionID partitionIDTarget;
        if (config.rle_length == -1 ||
            (config.evaluate && config.rle_length == -2)) {
          partitionIDTarget = (*config.stream_nodes_assign)[target];
        } else if (config.rle_length == 0) {
          partitionIDTarget = block_assignments->GetValueByIndex(target);
        } else {
          partitionIDTarget = block_assignments->GetValueByBatchIndex(target / config.rle_length, (target % config.rle_length));
        }

        if (partitionIDSource != partitionIDTarget) {
          edgeCut += edge_weight;
          (*stream_edges_assign)[node][partitionIDTarget] = true;
        }
      }
      (*lines)[0].clear();
      delete ss2;
      delete input;
      if (in.eof()) {
        break;
      }
    }
    edgeCut = edgeCut / 2; // Since every edge is counted twice
    qap = edgeCut;
    delete lines;

    std::vector<ImbalanceType> constraints_bound(config.number_of_constraints, 0);
    constraints_bound[0] = ceil(((100 + config.epsilon) / 100.) * (config.total_nodes/ (double)config.k));
    constraints_bound[1] = ceil(((100 + config.epsilon_edge) / 100.) * ((2*config.total_edges + config.total_nodes) / (double)config.k));
  
    std::vector<ImbalanceType> max(config.number_of_constraints, -1);
    std::vector<ImbalanceType> total_weight(config.number_of_constraints, 0);

    for (int i = 0; i < config.k; i++) {
        for(int j = 0; j < config.number_of_constraints; j++) {
          if (block_weights[i][j] > max[j]) {
              max[j] = block_weights[i][j];
          }
          if (block_weights[i][j] > constraints_bound[j]) {
          }
          total_weight[j] += block_weights[i][j];
        }
    }
    std::vector<ImbalanceType> perfect_balance(config.number_of_constraints, 0);
    perfect_balance[0] = ceil((config.total_nodes/ (double)config.k));
    perfect_balance[1] = ceil(((2*config.total_edges + config.total_nodes) / (double)config.k));
  
    for(int i = 0; i < config.number_of_constraints; i++) {
      balance[i] = max[i] / perfect_balance[i];
    }

    NodeID num_total_replicas = 0;  
    for (const auto& bitset : (*stream_edges_assign)) {
      num_total_replicas += static_cast<uint64_t>(bitset.count());
    }

    std::cout<<"num of replicas : "<< num_total_replicas - config.total_nodes<<std::endl;

    replication_factor = double(num_total_replicas)/double(config.total_nodes);

  }
}

void graph_io_stream::readOnePassPartitioning(  HeiClus::PartitionConfig &config, LongNodeID curr_node,
                                                int my_thread,
                                                std::vector<std::vector<LongNodeID>> *&input,
                                                const std::shared_ptr<CompressionDataStructure<PartitionID>>& block_assignments,
                                                vertex_partitioning *onepass_partitioner,
                                                std::vector<NodeWeight> & node_weights) {

  if(config.partitioning_type == NODE_PARTITIONING) {
    readNodeOnePassPartitioning(config, curr_node, my_thread, input, block_assignments, onepass_partitioner, node_weights);
  } else {
    readEdgeOnePassPartitioning(config, curr_node, my_thread, input, block_assignments, onepass_partitioner, node_weights);
  }
  
}

void graph_io_stream::readNodeOnePassPreAssignment(HeiClus::PartitionConfig &config, LongNodeID curr_node,
                                                  int my_thread,
                                                  std::vector<std::vector<LongNodeID>> *&input,
                                                  const std::shared_ptr<CompressionDataStructure<PartitionID>>& block_assignments,
                                                  vertex_partitioning *onepass_partitioner,
                                                  std::vector<NodeWeight> & node_weights,
                                                  std::vector<PartitionID> & com2part,
                                                  std::vector<PartitionID> & cluster_assignment ) {
  auto &read_ew = config.read_ew;
  auto &read_nw = config.read_nw;
  LongNodeID target;

  config.curr_node_neighbours.clear();

  LongNodeID cursor = (config.ram_stream) ? curr_node : 0;

  auto &all_blocks_to_keys = config.all_blocks_to_keys[my_thread];
  auto &next_key = config.next_key[my_thread];
  auto &neighbor_blocks = config.neighbor_blocks[my_thread];

  onepass_partitioner->clear_edgeweight_blocks(neighbor_blocks, next_key,
                                               my_thread);
  
  next_key = 0;

  std::vector<LongNodeID> &line_numbers = (*input)[cursor];
  LongNodeID col_counter = 0;
  node_weights[0] = (read_nw) ? line_numbers[col_counter++] : 1;
  /* total_nodeweight += weight; */

  PartitionID selecting_factor = (1 + (PartitionID)read_ew);
  config.edges = (line_numbers.size() - col_counter) / selecting_factor;

  config.curr_node_neighbours.reserve(config.edges);

  float scaling_factor = 1;
  LongNodeID deg = 0;
  while (col_counter < line_numbers.size()) {
    target = line_numbers[col_counter++];
    deg++;

    config.curr_node_neighbours.push_back(target - 1);

    EdgeWeight edge_weight = (read_ew) ? line_numbers[col_counter++] : 1;

    PartitionID targetGlobalPar, targetGlobalParTest;
    if (config.rle_length == -1) {
      targetGlobalPar = com2part[cluster_assignment[target - 1]];
    }
     
    if (targetGlobalPar != INVALID_PARTITION) {
      PartitionID key = all_blocks_to_keys[targetGlobalPar];
      if (key >= next_key || neighbor_blocks[key].first != targetGlobalPar) {
        all_blocks_to_keys[targetGlobalPar] = next_key;
        auto &new_element = neighbor_blocks[next_key];
        new_element.first = targetGlobalPar;
        new_element.second = edge_weight;
        next_key++;
      } else {
        neighbor_blocks[key].second += edge_weight;
      }
    }
  }

  node_weights[1] = deg + 1;

  if (deg > config.max_degree)
    config.max_degree = deg;

  for (PartitionID key = 0; key < next_key; key++) {
    auto &element = neighbor_blocks[key];
    onepass_partitioner->load_edge(element.first,
                                   element.second * scaling_factor, my_thread);
  }

  config.remaining_stream_nodes--;
}

void graph_io_stream::readEdgeOnePassPartitioning(HeiClus::PartitionConfig &config, LongNodeID curr_node,
                                                  int my_thread,
                                                  std::vector<std::vector<LongNodeID>> *&input,
                                                  const std::shared_ptr<CompressionDataStructure<PartitionID>>& block_assignments,
                                                  vertex_partitioning *onepass_partitioner,
                                                  std::vector<NodeWeight> & node_weights ) {
  auto &read_ew = config.read_ew;
  auto &read_nw = config.read_nw;
  LongNodeID target;

  LongNodeID cursor = (config.ram_stream) ? curr_node : 0;
  auto &neighbor_blocks = config.neighbor_blocks[my_thread];

  for (auto & neighbor_block : neighbor_blocks) {
    if(neighbor_block.first == -1) {
      break;
    }
    neighbor_block.first = -1;
    neighbor_block.second = 0;
  }
  
  std::vector<LongNodeID> &line_numbers = (*input)[cursor];
  LongNodeID col_counter = 0;
  /* total_nodeweight += weight; */

  PartitionID selecting_factor = (1 + (PartitionID)read_ew);
  config.edges = (line_numbers.size() - col_counter) / selecting_factor;
  LongNodeID deg = 0;

  if (neighbor_blocks.size() < config.edges) {
      neighbor_blocks.resize(config.edges, {-1,0});
  }

  NodeID neighbor_ix = 0;
  while (col_counter < line_numbers.size()) {
    target = line_numbers[col_counter++];
    deg++;
    EdgeWeight edge_weight = (read_ew) ? line_numbers[col_counter++] : 1;
    neighbor_blocks[neighbor_ix].first = target - 1;
    neighbor_blocks[neighbor_ix].second = edge_weight;
    neighbor_ix++;
  }

  if (deg > config.max_degree) {
    config.max_degree = deg;
  }

  // objective functions require degrees in edge partitioning.
  (*config.node_degrees)[curr_node] = deg;
  
  config.edges_streamed += deg;
}

void graph_io_stream::updateArtifWeight(HeiClus::PartitionConfig &config, 
                                        LongNodeID curr_node,
                                        int my_thread,
                                        std::vector<std::vector<LongNodeID>> *&input,
                                        vertex_partitioning *onepass_partitioner,
                                        std::vector<NodeWeight> & node_weights,
                                        PartitionID block) {
  
  auto &read_ew = config.read_ew;
  auto &read_nw = config.read_nw;
  LongNodeID target;

  LongNodeID cursor = (config.ram_stream) ? curr_node : 0;
  std::vector<LongNodeID> &line_numbers = (*input)[cursor];
  LongNodeID col_counter = 0;
  PartitionID selecting_factor = (1 + (PartitionID)read_ew);

  config.edges = (line_numbers.size() - col_counter) / selecting_factor;
  float scaling_factor = 1;
  LongNodeID deg = 0;

  (*config.stream_nodes_aw)[curr_node] = false;

  while (col_counter < line_numbers.size()) {
    target = line_numbers[col_counter++];
    deg++;
    EdgeWeight edge_weight = (read_ew) ? line_numbers[col_counter++] : 1;
    
    // Apply artificial weights for neighbouring future node that have not been assigned yet.
    if((target - 1) > curr_node 
        && (*config.stream_nodes_aw)[target - 1] == false
        && (*config.stream_nodes_assign)[target - 1] == -1) {

      (*config.stream_nodes_aw)[target - 1] = true;
      (*config.stream_nodes_assign)[target - 1] = block;
    }
  }
}
