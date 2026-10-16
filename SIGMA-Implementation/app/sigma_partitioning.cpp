#include <argtable3.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <regex.h>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <sys/resource.h>
#include <filesystem>
#include <vector>

#include "data_structure/ExternalPQ.h"
#include "data_structure/graph_access.h"
#include "graph_io_stream.h"
#include "macros_assertions.h"
#include "parse_parameters.h"
#include "partition/partition_config.h"
#include "quality_metrics.h"
#include "timer.h"
#include "tools/random_functions.h"

#include "partition/pre_partitioner/prepartition.h"
#include "partition/pre_partitioner/clustre_prepartition.h"

#include "partition/onepass_partitioning/fennel.h"
#include "partition/onepass_partitioning/fennel_range.h"
#include "partition/onepass_partitioning/fennel_multi_obj.h"
#include "partition/onepass_partitioning/hdrf.h"
#include "partition/onepass_partitioning/linear_obj_edge.h"
#include "partition/onepass_partitioning/vertex_partitioning.h"

#include "FlatBufferWriter.h"
#include "Stream_CPI_Info_generated.h"
#include "cpi/run_length_compression.hpp"

#include "data_structure/compression_vectors/CompressionDataStructure.h"
#include "data_structure/compression_vectors/RunLengthCompressionVector.h"
#include "data_structure/compression_vectors/BatchRunLengthCompression.h"

#define MIN(A, B) (((A) < (B)) ? (A) : (B))
#define MAX(A, B) (((A) > (B)) ? (A) : (B))

// Struct to store captured values
struct CapturedValues {
  std::size_t space_in_bytes;
  std::size_t uncompressed_space_in_bytes;
  double space_in_mib;
  double relative;
};

void initialize_onepass_partitioner(HeiClus::PartitionConfig &config,
                                    vertex_partitioning *&onepass_partitioner);

long getMaxRSS();

std::string extractBaseFilename(const std::string &fullPath);

std::ostream &cout_redirect();

CapturedValues parseCapturedValues(const std::string &output_str);

std::ostringstream redirected_cout;

// Helper: format a vector<T> as a JSON array on a single line
template <typename T>
static std::string vec_to_json(const std::vector<T> &v) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) oss << ", ";
    oss << v[i];
  }
  oss << "]";
  return oss.str();
}

void write_metrics_json(const std::string &path,
                        const std::string &dataset,
                        const std::string &partitioning_type,
                        int k,
                        LongNodeID num_nodes,
                        LongEdgeID num_edges,
                        EdgeWeight edge_cut,
                        double core_time,
                        const std::vector<ImbalanceType> &balance,
                        double replication_factor,
                        const std::vector<long long> &node_counts,
                        const std::vector<long long> &edge_counts,
                        long long num_unique_edges) {
  std::ofstream out(path);
  // Match the Python random partitioner's JSON layout/order as closely as possible.
  out << "{\n";
  out << "  \"algorithm\": \"multiconstraint\",\n";
  out << "  \"dataset\": \"" << dataset << "\",\n";
  out << "  \"k\": " << k << ",\n";
  out << "  \"partitioning_type\": \"" << partitioning_type << "\",\n";
  out << "  \"num_parts\": " << k << ",\n";
  out << "  \"num_nodes\": " << num_nodes << ",\n";
  out << "  \"num_edges\": " << num_edges << ",\n";

  if (partitioning_type == "edge") {
    // Edge partitioning: include num_unique_edges, replication_factor, balances; no edge_cut.
    out << "  \"num_unique_edges\": " << num_unique_edges << ",\n";
    out << "  \"node_counts\": " << vec_to_json(node_counts) << ",\n";
    out << "  \"edge_counts\": " << vec_to_json(edge_counts) << ",\n";

    const double total_nodes_in_parts = replication_factor * num_nodes;
    const double ceil_avg_node_load = std::ceil(total_nodes_in_parts / k);
    const double node_balance =
        total_nodes_in_parts > 0
            ? balance[0] * ceil_avg_node_load / (total_nodes_in_parts / k)
            : 0.0;

    out << "  \"edge_balance\": " << balance[1] << ",\n";
    out << "  \"node_balance\": " << node_balance << ",\n";
    out << "  \"replication_factor\": " << replication_factor << ",\n";
  } else {
    // Node partitioning: include edge_cut, edge_cut_ratio, balances.
    out << "  \"node_counts\": " << vec_to_json(node_counts) << ",\n";
    out << "  \"edge_counts\": " << vec_to_json(edge_counts) << ",\n";
    out << "  \"edge_cut\": " << edge_cut << ",\n";
    const double edge_cut_ratio =
        (num_edges > 0) ? (static_cast<double>(edge_cut) / static_cast<double>(num_edges)) : 0.0;
    out << "  \"edge_cut_ratio\": " << edge_cut_ratio << ",\n";
    out << "  \"node_balance\": " << balance[0] << ",\n";
    out << "  \"edge_balance\": " << balance[1] << ",\n";
  }
  out << "  \"core_time\": " << core_time << "\n";
  out << "}\n";
}

int main(int argn, char **argv) {
  timer e2e_t;
  e2e_t.restart();

  HeiClus::PartitionConfig config;
  std::string graph_filename;
  /* LINE_BUFFER lines = NULL; */
  std::vector<std::vector<LongNodeID>> *input = NULL;
  timer t, processing_t, io_t, clustre_t;
  EdgeID total_edge_cut = 0;
  int counter = 0;
  double global_mapping_time = 0;
  double buffer_mapping_time = 0;
  double buffer_io_time = 0;
  double total_time = 0;
  double ext_alg_time = 0;
  HeiClus::quality_metrics qm;
  EdgeWeight qap = 0;
  //double balance = 0;
  int full_stream_count = 0;
  double total_nodes = 0;

  LongNodeID num_lines = 1;
  int restreaming = 0;

  bool is_graph_weighted = false;
  bool suppress_output = false;
  bool recursive = false;

  int ret_code =
      parse_parameters(argn, argv, config, graph_filename, is_graph_weighted,
                       suppress_output, recursive);

  if (ret_code) {
    return 0;
  }

  std::streambuf *backup = std::cout.rdbuf();
  std::ofstream ofs;
  ofs.open("/dev/null");
  if (suppress_output) {
    std::cout.rdbuf(ofs.rdbuf());
  }
  std::string baseFilename = extractBaseFilename(graph_filename);

  srand(config.seed);
  HeiClus::random_functions::setSeed(config.seed);

  config.LogDump(stdout);
  config.stream_input = true;

  bool already_fully_partitioned;

  vertex_partitioning *onepass_partitioner = NULL;
  initialize_onepass_partitioner(config, onepass_partitioner);

  // container for storing block assignments used by Fennel
  std::shared_ptr<CompressionDataStructure<PartitionID>> block_assignments;

  int &passes = config.num_streams_passes;

  PrePartitionGraph * pre_partitioner = NULL;
  if(config.prepartition_graph == CLUSTRE) {
    clustre_t.restart();
    pre_partitioner = new PrePartitionGraph(config);
    HeiClus::PartitionConfig pre_partition_config = pre_partitioner->start_prepartitioning();

    if(config.partitioning_type == NODE_PARTITIONING) {
      config.stream_nodes_assign = pre_partition_config.stream_nodes_assign;

      if(config.one_pass_algorithm == ONEPASS_FENNEL_MULT_OBJ) {
        config.stream_edges_assign = pre_partition_config.stream_edges_assign;
      }

    } else {
      config.stream_edges_assign = pre_partition_config.stream_edges_assign;
      config.node_degrees = pre_partition_config.node_degrees;
      config.stream_nodes_assign = pre_partition_config.stream_nodes_assign;
    }
    config.stream_blocks_weight = pre_partition_config.stream_blocks_weight;
    config.stream_blocks_load = pre_partition_config.stream_blocks_load;

    ext_alg_time = clustre_t.elapsed();
    total_time += ext_alg_time;
  }

  NodeID num_preassigned_nodes = 0;
  EdgeID num_preassigned_edges = 0;
  for(PartitionID part = 0; part < config.k; part++) {
      num_preassigned_nodes += (*config.stream_blocks_load)[part][0];
      num_preassigned_edges += (*config.stream_blocks_load)[part][1];
  }

  std::ofstream part_edge_file;
  if (!config.suppress_file_output){
    std::stringstream filename_edge;
    if (!config.filename_output.compare("")) {
        filename_edge << baseFilename << "_" << config.k;
    } else {
        filename_edge << config.filename_output;
    }
    part_edge_file.open(filename_edge.str());
  }

  for (config.restream_number = 0; config.restream_number < passes; config.restream_number++) {

    processing_t.restart();

    io_t.restart();
    graph_io_stream::readFirstLineStreamPartitioning(config, graph_filename, total_edge_cut, qap);
    buffer_io_time += io_t.elapsed();

    // set up block assignment container based on algorithm configuration
    if(config.rle_length == 0) {
        block_assignments = std::make_shared<RunLengthCompressionVector<PartitionID>>();
    }
    else if (config.rle_length > 0) {
        block_assignments = std::make_shared<BatchRunLengthCompression<PartitionID>>((config.total_nodes /
                                                                                        config.rle_length) + 1);
    }

    onepass_partitioner->instantiate_blocks(config.remaining_stream_nodes,
                                            config.remaining_stream_edges,
                                            config.k,
                                            config.number_of_constraints,
                                            config.imbalance,
                                            config.epsilon_edge,
                                            config);

    for (int i = 0; i < config.parallel_nodes; i++) {
      config.all_blocks_to_keys[i].resize(config.k);
      for (auto &b : config.all_blocks_to_keys[i]) {
        b = INVALID_PARTITION;
      }
      config.neighbor_blocks[i].resize(config.k);
      config.next_key[i] = 0;
    }

    if(config.prepartition_graph == CLUSTRE) {
      for(int part = 0; part < (*config.stream_blocks_load).size(); part++) {
        std::vector<NodeWeight> & block_weight = (*config.stream_blocks_load)[part];
        if(block_weight[0] > 0) {
          onepass_partitioner->set_decision_partitioning(part, -1, block_weight, 0);
        }

        if(block_weight[0] > onepass_partitioner->blocks[config.max_load_nodes_part_hdrf].partition_weights[0]) {
          config.max_load_nodes_part_hdrf = part;
          if(config.capacity_min_cap < onepass_partitioner->blocks[part].partition_weights[0] / onepass_partitioner->blocks[part].partition_constraints[0]) {
            config.capacity_min_cap = static_cast<double>(onepass_partitioner->blocks[part].partition_weights[0] / onepass_partitioner->blocks[part].partition_constraints[0]);            
          }
        }

        if(block_weight[1] > onepass_partitioner->blocks[config.max_load_edges_part_hdrf].partition_weights[1]) {
          config.max_load_edges_part_hdrf = part;
          if(config.capacity_min_cap < onepass_partitioner->blocks[part].partition_weights[1] / onepass_partitioner->blocks[part].partition_constraints[1]) {
            config.capacity_min_cap = static_cast<double>(onepass_partitioner->blocks[part].partition_weights[1] / onepass_partitioner->blocks[part].partition_constraints[1]);            
          }
        }
      }
    }

    LongNodeID node_capacity_batch_size = ceil( (config.total_nodes / ( config.num_capacity_changes ) ) );
    if(config.partitioning_type == EDGE_PARTITIONING) {
      LongNodeID node_capacity_batch_size = INFINITY;
    }
    EdgeID edge_capacity_batch_size = ceil( ( ( 2 * config.total_edges + config.total_nodes ) / config.num_capacity_changes ));

    for (LongNodeID curr_node = 0; curr_node < config.n_batches; curr_node++) {
      int my_thread = 0;
      std::vector<NodeWeight> node_weights(config.number_of_constraints, 0);

//     if(config.use_dynamic_capacity && config.partitioning_type == NODE_PARTITIONING
//         // the 1.2 signifies that we will force upto 20% of nodes to be in a non optimal partition, due to the capacity constraint
//         // and therefore distribute the nodes and edges more evenly throughout the partitioning
//         && ( curr_node % node_capacity_batch_size == 0  || config.edges_streamed > edge_capacity_batch_size)
//         ) {
//
//         std::cout<<"using dynamic capacity"<<std::endl;
//
//         onepass_partitioner->adjust_capacity(curr_node, config);
//
//         if(config.edges_streamed > edge_capacity_batch_size) {
//             edge_capacity_batch_size += edge_capacity_batch_size;
//         }
//     }

      io_t.restart();
      graph_io_stream::loadBufferLinesToBinary(config, input, num_lines, curr_node, restreaming);
      buffer_io_time += io_t.elapsed();

      if(config.partitioning_type == NODE_PARTITIONING) {
        if((*config.stream_nodes_assign)[curr_node] != -1
          && ((config.with_aw && (*config.stream_nodes_aw)[curr_node] == false) || !config.with_aw)) {
          delete input;
          continue;
        }
      }

      t.restart();
      graph_io_stream::readOnePassPartitioning(config, curr_node, my_thread, input, block_assignments, onepass_partitioner, node_weights);

      // NEW
      config.stream_progress = (config.n_batches > 1)
          ? static_cast<double>(curr_node) / static_cast<double>(config.n_batches - 1)
          : 1.0;

      PartitionID block = onepass_partitioner->assign_to_partition(
          curr_node, node_weights, config.previous_assignment, config, config.kappa, my_thread, part_edge_file);

      if(config.with_aw && config.partitioning_type == NODE_PARTITIONING) {
        graph_io_stream::updateArtifWeight(config, curr_node, my_thread, input, onepass_partitioner, node_weights, block);
      }

      if(config.partitioning_type == NODE_PARTITIONING) {
        if (config.rle_length == -1) {
          (*config.stream_nodes_assign)[curr_node] = block;
        } else if (config.rle_length == 0) {
          block_assignments->Append(block);
        }

        config.previous_assignment = block;
        (*config.stream_blocks_weight)[block] += 1;

        for(int i = 0; i < config.number_of_constraints; i++) {
          (*config.stream_blocks_load)[block][i] += node_weights[i];
        }
      }

      if (!config.ram_stream) {
        delete input;
      }

      global_mapping_time += t.elapsed();
    }

    total_time += processing_t.elapsed();

    if (config.ram_stream) {
      delete input;
      /* delete lines; */
    }
  }

  if(!config.suppress_file_output && config.partitioning_type == EDGE_PARTITIONING) {
    // Close the print stream of edges
    part_edge_file.close();
  }

  long overall_max_RSS = getMaxRSS();

  std::vector<ImbalanceType> balance(config.number_of_constraints, 0);

  double replication_factor = 0;

  if (config.write_results) {
      if (!config.suppress_output) {
          if (((config.one_pass_algorithm == ONEPASS_HASHING) ||
               (config.one_pass_algorithm == ONEPASS_HASHING_CRC32)) && !config.evaluate) {
              total_edge_cut = 0;
          } else {
              if (config.rle_length != -2 || config.evaluate) {
                  graph_io_stream::streamEvaluatePartition(config, graph_filename,
                                                          total_edge_cut, qap,
                                                          block_assignments, balance, replication_factor);
              }
          }
            // balance = qm.balance_full_stream(*config.stream_blocks_load);
      }

      CapturedValues capturedValues;
      if (config.rle_length == -1 && ((config.one_pass_algorithm != ONEPASS_HASHING) &&
                                      (config.one_pass_algorithm != ONEPASS_HASHING_CRC32))) {
        std::streambuf *original_cout_buffer = std::cout.rdbuf();
        std::cout.rdbuf(redirected_cout.rdbuf());
        std::cout << "Performing RLE compression test..." << std::endl;
        cpi::RunLengthCompression rlc(*config.stream_nodes_assign);
        auto p_id = rlc[42];    // access partition ids;
        rlc.print_statistics(); // print statistics
        std::string output_str = redirected_cout.str();
        capturedValues = parseCapturedValues(output_str);
        std::cout.rdbuf(original_cout_buffer);
      } else {
        capturedValues.space_in_bytes = 0;
        capturedValues.uncompressed_space_in_bytes = 0;
        capturedValues.space_in_mib = 0;
        capturedValues.relative = 0;
      }

      FlatBufferWriter fb_writer;
      fb_writer.updateResourceConsumption(buffer_io_time, ext_alg_time, global_mapping_time,
                                        total_time, overall_max_RSS);
      fb_writer.updatePartitionMetrics(total_edge_cut, replication_factor, balance);
      fb_writer.updateCompressionStatistics(
        capturedValues.space_in_bytes,
        capturedValues.uncompressed_space_in_bytes, capturedValues.space_in_mib,
        capturedValues.relative);
      fb_writer.writePartitioning(baseFilename, config);
  }

  std::vector<long long> node_counts(config.k, 0);
  std::vector<long long> edge_counts(config.k, 0);
  long long num_unique_edges = 0;

  if (config.partitioning_type == NODE_PARTITIONING) {
    if (config.stream_blocks_weight != nullptr) {
      for (int p = 0; p < config.k && p < (int)(*config.stream_blocks_weight).size(); ++p) {
        node_counts[p] = static_cast<long long>((*config.stream_blocks_weight)[p]);
      }
    }
    if (config.stream_blocks_load != nullptr) {
      // For node partitioning, the second constraint slot holds incident-edge
      // load per block (matches how `balance[1]` / edge_balance is computed).
      for (int p = 0; p < config.k && p < (int)(*config.stream_blocks_load).size(); ++p) {
        const auto &load = (*config.stream_blocks_load)[p];
        if (load.size() >= 2) {
          edge_counts[p] = static_cast<long long>(load[1]);
        }
      }
    }
  } else {
    // Edge partitioning
    if (config.stream_blocks_load != nullptr) {
      for (int p = 0; p < config.k && p < (int)(*config.stream_blocks_load).size(); ++p) {
        const auto &load = (*config.stream_blocks_load)[p];
        // load[0] = node endpoints assigned to this part (with replication)
        // load[1] = edges assigned to this part
        if (load.size() >= 1) node_counts[p] = static_cast<long long>(load[0]);
        if (load.size() >= 2) edge_counts[p] = static_cast<long long>(load[1]);
      }
    }
    // Streaming edge partitioning operates on the (already-undirected) edge
    // stream, so total partitioned edges == config.total_edges.
    num_unique_edges = static_cast<long long>(config.total_edges);
  }

  double e2e_time = e2e_t.elapsed();
  if (!config.suppress_output) {
    std::cout << "CODEx_METRICS"
              << " algorithm=multiconstraint"
              << " dataset=" << baseFilename
              << " partitioning_type="
              << ((config.partitioning_type == EDGE_PARTITIONING) ? "edge" : "node")
              << " k=" << config.k
              << " num_nodes=" << config.total_nodes
              << " num_edges=" << config.total_edges
              << " core_time=" << total_time
              << " e2e_time=" << e2e_time;

    if (config.partitioning_type == EDGE_PARTITIONING) {
      std::cout << " node_balance=" << balance[0]
                << " edge_balance=" << balance[1]
                << " replication_factor=" << replication_factor;
    } else {
      const double edge_cut_ratio =
          (config.total_edges > 0)
              ? (static_cast<double>(total_edge_cut) /
                 static_cast<double>(config.total_edges))
              : 0.0;
      std::cout << " edge_cut=" << total_edge_cut
                << " edge_cut_ratio=" << edge_cut_ratio
                << " node_balance=" << balance[0]
                << " edge_balance=" << balance[1];
    }

    std::cout << std::endl;
  }

  if (!config.output_path.empty()) {
    std::filesystem::create_directories(config.output_path);
  }
  write_metrics_json(
      config.output_path + baseFilename + "_metrics.json",
      baseFilename,                               // dataset = base filename (matches Python default)
      (config.partitioning_type == EDGE_PARTITIONING) ? "edge" : "node",
      config.k,
      config.total_nodes,
      config.total_edges,
      total_edge_cut,
      total_time,
//      e2e_time,                                   // proper end-to-end time
      balance,
      replication_factor,
      node_counts,
      edge_counts,
      num_unique_edges);

    // write the partition to the disc
    std::stringstream filename;
    if (!config.filename_output.compare("")) {
        filename << baseFilename << "_" << config.k;
    } else {
        filename << config.filename_output;
    }

    if (!config.suppress_file_output) {
        if (config.rle_length != -2 || config.evaluate) {
            graph_io_stream::writePartitionStream(config, filename.str(), block_assignments);
        }
    } else {
        std::cout << "No partition will be written as output." << std::endl;
    }

  return 0;
}

void initialize_onepass_partitioner(HeiClus::PartitionConfig &config,
                                    vertex_partitioning *&onepass_partitioner) {
  switch (config.one_pass_algorithm) {
    case ONEPASS_RANGE_FENNEL:
    onepass_partitioner = new onepass_range_fennel(0, config.k - 1, config.stream_rec_bisection_base,
                           config.parallel_nodes, false, config.fennel_gamma);
    break;

    // case ONEPASS_FENNEL:
    // onepass_partitioner =
    //     new onepass_fennel(0, config.k - 1, config.stream_rec_bisection_base,
    //                        config.parallel_nodes, false, config.fennel_gamma);
    // break;

    case ONEPASS_HDRF:
    onepass_partitioner =
        new onepass_hdrf(0, config.k - 1, config.stream_rec_bisection_base,
                           config.parallel_nodes, false, config.fennel_gamma);
    break;

    // case ONEPASS_EDGES_LINEAR:
    // onepass_partitioner =
    //     new onepass_linear_obj_edge(0, config.k - 1, config.stream_rec_bisection_base,
    //                        config.parallel_nodes, false, config.fennel_gamma);
    // break;

    case ONEPASS_FENNEL_MULT_OBJ:
    onepass_partitioner =
        new onepass_fennel_multi_obj(0, config.k - 1, config.stream_rec_bisection_base,
                           config.parallel_nodes, false, config.fennel_gamma);
    break;
  }
}

long getMaxRSS() {
  struct rusage usage;

  if (getrusage(RUSAGE_SELF, &usage) == 0) {
    // The maximum resident set size is in kilobytes
    return usage.ru_maxrss;
  } else {
    std::cerr << "Error getting resource usage information." << std::endl;
    // Return a sentinel value or handle the error in an appropriate way
    return -1;
  }
}

// Redirect cout to the stringstream
std::ostream &cout_redirect() {
  static std::ostream cout_redirector(redirected_cout.rdbuf());
  return cout_redirector;
}

// Function to parse the captured values from the redirected output
CapturedValues parseCapturedValues(const std::string &output_str) {
  CapturedValues values;

  // Extract values using stream extraction
  std::istringstream stream(output_str);

  // Ignore text up to '=' and then extract values
  stream.ignore(std::numeric_limits<std::streamsize>::max(), '=');
  stream >> values.space_in_bytes;

  stream.ignore(std::numeric_limits<std::streamsize>::max(), '=');
  stream >> values.uncompressed_space_in_bytes;

  stream.ignore(std::numeric_limits<std::streamsize>::max(), '=');
  stream >> values.space_in_mib;

  stream.ignore(std::numeric_limits<std::streamsize>::max(), '=');
  stream >> values.relative;

  return values;
}

// Function to extract the base filename without path and extension
std::string extractBaseFilename(const std::string &fullPath) {
  size_t lastSlash = fullPath.find_last_of('/');
  size_t lastDot = fullPath.find_last_of('.');

  if (lastSlash != std::string::npos) {
    // Found a slash, extract the substring after the last slash
    return fullPath.substr(lastSlash + 1, lastDot - lastSlash - 1);
  } else {
    // No slash found, just extract the substring before the last dot
    return fullPath.substr(0, lastDot);
  }
}
