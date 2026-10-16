#include <bits/stdc++.h>
#include <sys/stat.h>
#define DB(x) cerr << __LINE__ << ": " << #x << " = " << (x) << endl
using namespace std;
using namespace std::chrono;

// Number of partitions
// OLD:
// const int k = 2;
// const int mx = 1000005;
// set<int> partitions[k];
// set<int> adjacency[mx];

// NEW:
int k = 2;
vector<set<int>> partitions;       // sized later: partitions.resize(k)
vector<set<int>> adjacency;        // sized later: adjacency.assign(n+1, set<int>())


// Parameters
double alpha = 0.5;
double gammaPower = 1.5;

double partitionCost(double sz) {
    double cost = 0;
    cost = alpha * pow(sz+1.0, gammaPower) - alpha * pow(sz, gammaPower);
    return cost;
}

double Cost(double sz) {
    double cost = 0;
    cost = alpha * pow(sz, gammaPower);
    return cost;
}

namespace {
string basename_of(const string& path) {
    size_t pos = path.find_last_of("/\\");
    return pos == string::npos ? path : path.substr(pos + 1);
}

void ensure_dir_exists(const string& path) {
#ifdef _WIN32
    _mkdir(path.c_str());
#else
    mkdir(path.c_str(), 0755);
#endif
}

void write_node_metrics_json(const string& artifact_dir,
                             const string& dataset,
                             int num_nodes,
                             int num_edges,
                             int edge_cut,
                             double edge_cut_ratio,
                             double node_balance,
                             double edge_balance,
                             double core_time) {
    ensure_dir_exists(artifact_dir);
    ofstream metrics(artifact_dir + "/metrics.json");
    metrics << "{\n";
    metrics << "  \"algorithm\": \"fennel\",\n";
    metrics << "  \"dataset\": \"" << dataset << "\",\n";
    metrics << "  \"k\": " << k << ",\n";
    metrics << "  \"partitioning_type\": \"node\",\n";
    metrics << "  \"num_parts\": " << k << ",\n";
    metrics << "  \"num_nodes\": " << num_nodes << ",\n";
    metrics << "  \"num_edges\": " << num_edges << ",\n";
    metrics << "  \"edge_cut\": " << edge_cut << ",\n";
    metrics << "  \"edge_cut_ratio\": " << edge_cut_ratio << ",\n";
    metrics << "  \"node_balance\": " << node_balance << ",\n";
    metrics << "  \"edge_balance\": " << edge_balance << ",\n";
    metrics << "  \"core_time\": " << core_time << "\n";
    metrics << "}\n";
}
}

int main(int argc, char** argv) {
    auto main_start = high_resolution_clock::now();
    srand(42);

    // Parse command-line arguments: ./fennel <input_file> <k>
    string input_path = "stdin";
    if (argc > 1) {
        input_path = argv[1];
        if (!freopen(argv[1], "r", stdin)) {
            cerr << "Error: cannot open input file " << argv[1] << "\n";
            return 1;
        }
    }
    if (argc > 2) {
        k = atoi(argv[2]);
        if (k <= 0) {
            cerr << "Error: k must be positive\n";
            return 1;
        }
    }
    string output_path = "";  // empty means: write to stdout
    if (argc > 3) {
        output_path = argv[3];
    }
    partitions.assign(k, set<int>());

    int n, m = 0;
    cin >> n;
    adjacency.assign(n + 1, set<int>());   // dynamic sizing instead of fixed mx
    for(int i = 1; i <= n; ++i) {
        int number_of_nodes;
        cin >> number_of_nodes;
        m += number_of_nodes;
        for(int j = 0; j < number_of_nodes; ++j) {
            int x;
            cin >> x;
            adjacency[i].insert(x);
        }
    }
    m /= 2;

    if (k > n) {
        cerr << "Error: k (" << k << ") cannot exceed number of nodes (" << n << ")\n";
        return 1;
    }

    // Results
    vector<int> cutEdge(k, 0);
    vector<int> partition_of_node(n + 1, -1);

    // Ordering of streaming vertices
    vector<int> ordering(n);
    for(int i = 1; i <= n; ++i) {
        ordering[i-1] = i;
    }
    random_shuffle(ordering.begin(), ordering.end());

    // Initial paritions
    for(int i = 0; i < k; ++i) {
        partitions[i].insert(ordering[i]);
        partition_of_node[ordering[i]] = i;
    }

    auto start = high_resolution_clock::now();

    // Streaming vertices
    for(int node_number = k; node_number < n; ++node_number) {
        int finalPartition = 0, node = ordering[node_number], additionalEdge = 0;
        double objectiveFunctionScore = -1e18;
        for(int container = 0; container < k; ++container) {
            double intraPartition = partitionCost((int)partitions[container].size());
            int interPartition = 0;

            for(auto neighbours: adjacency[node]) {
                if(partitions[container].find(neighbours) != partitions[container].end()) {
                    ++interPartition;
                }
            }
            // DB(interPartition);
            // DB(intraPartition);
            if(((double)interPartition) - intraPartition > objectiveFunctionScore) {
                objectiveFunctionScore = interPartition - intraPartition;
                finalPartition = container;
                additionalEdge = interPartition;
            }
        }
        partitions[finalPartition].insert(node);
        partition_of_node[node] = finalPartition;
        cutEdge[finalPartition] += additionalEdge;
    }


//    for(int i = 0; i < k; ++i) {
//        cout << "Partition: " << i+1 << "\n";
//        for(auto it: partitions[i]) cout << it << " ";
//        cout << "\n---\n";
//    }

    // partition_of_node[i] contains the partition of node i (0-based)
    if (output_path.empty()) {
        for (int i = 1; i <= n; ++i) {
            cout << partition_of_node[i] << "\n";
        }
    } else {
        ofstream out(output_path);
        if (!out) {
            cerr << "Error: cannot open output file " << output_path << "\n";
            return 1;
        }
        for (int i = 1; i <= n; ++i) {
            out << partition_of_node[i] << "\n";
        }
    }

    auto stop = high_resolution_clock::now();

    double result = 0;
    int totalCutEdges = 0;
    for(int i = 0; i < k; ++i) {
        DB(cutEdge[i]);
        DB(partitionCost(partitions[i].size()));
        result += ((double) cutEdge[i]) - Cost(partitions[i].size());
        totalCutEdges += cutEdge[i];
    }
    totalCutEdges = m - totalCutEdges;
    // DB(totalCutEdges);
    DB(result);
    auto duration = duration_cast<microseconds>(stop - start);
    DB(duration.count());

    vector<int> part_node_counts(k, 0);
    vector<size_t> part_edge_loads(k, 0);
    size_t total_degree = 0;
    int edge_cut = 0;
    for(int node = 1; node <= n; ++node) {
        int partition = partition_of_node[node];
        if(partition < 0) {
            continue;
        }
        part_node_counts[partition]++;
        part_edge_loads[partition] += adjacency[node].size();
        total_degree += adjacency[node].size();
        for(int neighbour : adjacency[node]) {
            if(node < neighbour && partition_of_node[neighbour] != partition) {
                ++edge_cut;
            }
        }
    }

    int max_nodes = 0;
    size_t max_edge_load = 0;
    for(int count : part_node_counts) {
        max_nodes = max(max_nodes, count);
    }
    for(size_t edge_load : part_edge_loads) {
        max_edge_load = max(max_edge_load, edge_load);
    }

    double node_balance = static_cast<double>(max_nodes) / ceil(static_cast<double>(n) / k);
    double edge_balance = total_degree == 0 ? 0.0 : static_cast<double>(max_edge_load) / ceil(static_cast<double>(total_degree) / k);
    string dataset = (argc > 1) ? basename_of(input_path) : "stdin";
    double core_time =
        duration_cast<microseconds>(high_resolution_clock::now() - main_start).count() / 1000000.0;
    write_node_metrics_json("fennel.artifacts",
                            dataset,
                            n,
                            m,
                            edge_cut,
                            m == 0 ? 0.0 : static_cast<double>(edge_cut) / m,
                            node_balance,
                            edge_balance,
                            core_time);

    return 0;
}
