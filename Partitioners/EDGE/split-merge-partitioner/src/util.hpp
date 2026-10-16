#ifndef UTIL_HPP
#define UTIL_HPP

#include <chrono>
#include <filesystem>
#include <algorithm>

#include "common.hpp"

inline std::string h2hedgelist_name(const std::string &basefilename)
{
	return basefilename + ".h2h_edgelist";
}

inline std::string lowedgelist_name(const std::string &basefilename)
{
	return basefilename + ".low_edgelist";
}

inline std::string binedgelist_name(const std::string &basefilename)
{
    return basefilename + ".binedgelist";
}

inline std::string degree_name(const std::string &basefilename)
{
    return basefilename + ".degree";
}

inline std::string edge_partitioned_name(const std::string &basefilename)
{
    std::string ret = basefilename + ".edgepart.";
    if (FLAGS_method.substr(0, 3) == "fsm") {
        std::string split_method = FLAGS_method == "fsm" ? "ne" : FLAGS_method.substr(4);
        if (FLAGS_k == 1) {
            ret += split_method;
            if (split_method == "hep") {
                ret += "_hdf_" + std::to_string(int(FLAGS_hdf));
            }
        } else {
            ret += "fsm_" + split_method + "_k_" + std::to_string(FLAGS_k);
        }
    } else if (FLAGS_method == "hep") {
        ret += "hep_hdf_" + std::to_string((int)FLAGS_hdf);
    } else if (FLAGS_method.substr(0, 3) == "v2e") {
        std::string split_method = FLAGS_method.substr(4);
        if (FLAGS_k == 1) {
            ret += split_method;
        } else {
            ret += "fsm_" + split_method + "_k_" + std::to_string(FLAGS_k);
        }
    } else {
        ret += FLAGS_method;
    }
    ret += "." + std::to_string(FLAGS_p);
    return ret;
}

inline std::string vertex_partitioned_name(const std::string &basefilename)
{
    std::string ret = basefilename + ".vertexpart.";
    ret += FLAGS_method + ".";
    ret += std::to_string(FLAGS_p);
    return ret;
}

inline std::string metrics_name(const std::string &basefilename)
{
    const std::string metrics_dir = edge_partitioned_name(basefilename) + ".artifacts";
    std::filesystem::create_directories(metrics_dir);
    return metrics_dir + "/metrics.json";
}

inline std::chrono::steady_clock::time_point& program_timer_start()
{
    static std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    return start;
}

inline void set_program_timer_start()
{
    program_timer_start() = std::chrono::steady_clock::now();
}

inline double elapsed_since_program_start()
{
    // Repository-wide convention: measure until the final metrics write.
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - program_timer_start())
        .count();
}

template <typename TBitset>
inline void write_edge_metrics_json(
    const std::string &basefilename,
    const std::string &algorithm,
    bid_t num_partitions,
    vid_t num_vertices,
    eid_t num_edges,
    const std::vector<eid_t> &occupied,
    const std::vector<TBitset> &is_boundarys,
    double core_time,
    double e2e_time)
{
    (void)core_time;
    (void)e2e_time;
    std::vector<vid_t> node_counts(num_partitions, 0);
    vid_t total_mirrors = 0;
    for (bid_t b = 0; b < num_partitions; ++b) {
        node_counts[b] = is_boundarys[b].popcount();
        total_mirrors += node_counts[b];
    }

    const auto max_nodes = *std::max_element(node_counts.begin(), node_counts.end());
    const auto max_occupied = *std::max_element(occupied.begin(), occupied.end());

    std::ofstream metrics_file(metrics_name(basefilename));
    metrics_file << "{\n";
    metrics_file << "  \"algorithm\": \"" << algorithm << "\",\n";
    metrics_file << "  \"dataset\": \"" << basefilename << "\",\n";
    metrics_file << "  \"k\": " << static_cast<unsigned int>(num_partitions) << ",\n";
    metrics_file << "  \"partitioning_type\": \"edge\",\n";
    metrics_file << "  \"num_parts\": " << static_cast<unsigned int>(num_partitions) << ",\n";
    metrics_file << "  \"num_nodes\": " << num_vertices << ",\n";
    metrics_file << "  \"num_edges\": " << num_edges << ",\n";
    metrics_file << "  \"node_balance\": "
                 << (double)max_nodes / ((double)total_mirrors / num_partitions) << ",\n";
    metrics_file << "  \"edge_balance\": "
                 << (double)max_occupied / ((double)num_edges / num_partitions) << ",\n";
    metrics_file << "  \"replication_factor\": "
                 << (double)total_mirrors / num_vertices << ",\n";
    metrics_file << "  \"core_time\": " << elapsed_since_program_start() << "\n";
    metrics_file << "}\n";
}

inline bool is_exists(const std::string &name)
{
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

class Timer
{
  private:
    std::chrono::system_clock::time_point t1, t2;
    double total;

  public:
    Timer() : total(0) {}
    void reset() { total = 0; }
    void start() { t1 = std::chrono::system_clock::now(); }
    void stop()
    {
        t2 = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = t2 - t1;
        total += diff.count();
    }
    double get_time() { return total; }
};

template <typename T>
double jains_fairness(const std::vector<T>& L)
{
    double a = 0.0;
    for (const auto& x : L) {
        a += static_cast<double>(x);
    }

    double b = 0.0;
    for (const auto& x : L) {
        b += static_cast<double>(x) * static_cast<double>(x);
    }
    b *= static_cast<double>(L.size());

    return (a * a) / b;
}

#endif
