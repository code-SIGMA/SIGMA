#include <vector>
#include <string>
#include <filesystem>
#include <chrono>

#include "globals.hpp"
#include "stats.hpp"
#include "dbh.hpp"
#include "../converter/conversions.hpp"

DEFINE_string(filename, "", "name of the file to store edge list of a graph.");
DEFINE_int32(p, 0, "the number of partitions that the graph will be divided into.");
DEFINE_uint64(memsize, 100, "memory size in MB. "
                            "Memsize is used as a chunk size if shuffling and batch size while partitioning");
DEFINE_bool(write_parts, false, "write out the partitions to a file");
DEFINE_string(parts_filename, "", "partitions filename");
DEFINE_string(output_path, "", "results output path");


double start_partitioning(Globals& globals, DBH& partitioner, std::string& binedgelist);

int main(int argc, char *argv[]) 
{
    const auto main_start = std::chrono::steady_clock::now();

    // set up glogs and gflags
    google::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;
    google::HandleCommandLineHelpFlags();

    // convert edge list to binary format
    Converter* converter;
    std::string binedgelist;
    converter = new Converter(FLAGS_filename);
    binedgelist = binedgelist_name(FLAGS_filename);
    convert(FLAGS_filename, converter);

    // set up globals
    std::ifstream fin(binedgelist, std::ios::binary | std::ios::ate);
    Globals globals(fin, FLAGS_filename, FLAGS_p);

    // init partitioner
    DBH dbh(globals);

    // start partitioner
    start_partitioning(globals, dbh, binedgelist);

    Stats stats(dbh, globals);
    stats.compute_and_print_stats();
    if (!FLAGS_output_path.empty())
    {
        const double core_time =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - main_start).count();
        std::filesystem::create_directories(FLAGS_output_path);
        std::ofstream metrics_file(FLAGS_output_path + "/metrics.json");
        metrics_file << "{\n";
        metrics_file << "  \"algorithm\": \"2ps_base_dbh\",\n";
        metrics_file << "  \"dataset\": \"" << FLAGS_filename << "\",\n";
        metrics_file << "  \"k\": " << globals.NUM_PARTITIONS << ",\n";
        metrics_file << "  \"partitioning_type\": \"edge\",\n";
        metrics_file << "  \"num_parts\": " << globals.NUM_PARTITIONS << ",\n";
        metrics_file << "  \"num_nodes\": " << globals.NUM_VERTICES << ",\n";
        metrics_file << "  \"num_edges\": " << globals.NUM_EDGES << ",\n";
        metrics_file << "  \"node_balance\": " << stats.get_node_balance() << ",\n";
        metrics_file << "  \"edge_balance\": " << stats.get_edge_balance() << ",\n";
        metrics_file << "  \"replication_factor\": " << stats.get_replication_factor() << ",\n";
        metrics_file << "  \"core_time\": " << core_time << "\n";
        metrics_file << "}\n";
    }

  return 0;
};

double start_partitioning(Globals& globals, DBH& partitioner, std::string& binedgelist)
{
    Timer partitioner_timer;
    partitioner_timer.start();

    globals.calculate_degrees();
    partitioner.perform_partitioning();

    partitioner_timer.stop();
    LOG(INFO) << "partitioning time: " << partitioner_timer.get_time();
    return partitioner_timer.get_time();
}
