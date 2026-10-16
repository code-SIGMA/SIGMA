/*
//@HEADER
// *****************************************************************************
//
// PULP: Multi-Objective Multi-Constraint Partitioning Using Label Propagation
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions?  Contact  George M. Slota (gmslota@sandia.gov)
//                      Siva Rajamanickam (srajama@sandia.gov)
//
// *****************************************************************************
//@HEADER
*/

using namespace std;

#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <math.h>
#include <omp.h>
#include <sstream>
#include <string.h>
#include <string>
#include <vector>

#include "io.cpp"
#include "pulp.h"

void print_usage_full(char **argv) {
    printf("To run: %s [graphfile] [num parts] [options]\n\n", argv[0]);
    printf("Options:\n");
    printf("\t-v [#.#]:\n");
    printf("\t\tVertex balance constraint [default: 1.10 (10%%)]\n");
    printf("\t-e [#.#]:\n");
    printf("\t\tEdge balance constraint [default: off]\n");
    printf("\t-c:\n");
    printf("\t\tAttempt to minimize per-part cut\n");
    printf("\t-l:\n");
    printf("\t\tDo label propagation-based initialization\n");
    printf("\t-m [#]:\n");
    printf("\t\tGenerate multiple partitions [default: 1]\n");
    printf("\t-o [file]:\n");
    printf("\t\tOutput parts file [default: graphname.part.numparts]\n");
    printf("\t-i [file]:\n");
    printf("\t\tInput parts file [default: none]\n");
    printf("\t-s [seed]:\n");
    printf("\t\tSet seed integer [default: random int]\n");
    printf("\t-q:\n");
    printf("\t\tEvaluate partition quality (verbose stats)\n");
    printf("\t-n [name]:\n");
    printf("\t\tDataset name used in the metrics output\n");
    printf("\t\t[default: graph filename without extension]\n");
    printf("\t-M [file]:\n");
    printf("\t\tMetrics JSON output file\n");
    printf("\t\t[default: metrics.json next to the parts file]\n");
    exit(0);
}

/*
'##::::'##:'########:'########:'########::'####:'########::'######::
 ###::'###: ##.....::... ##..:: ##.... ##:. ##:: ##.... ##:'##... ##:
 ####'####: ##:::::::::: ##:::: ##:::: ##:: ##:: ##:::: ##: ##:::..::
 ## ### ##: ######:::::: ##:::: ########::: ##:: ##:::: ##:. ######::
 ##. #: ##: ##...::::::: ##:::: ##.. ##:::: ##:: ##:::: ##::..... ##:
 ##:.:: ##: ##:::::::::: ##:::: ##::. ##::: ##:: ##:::: ##:'##::: ##:
 ##:::: ##: ########:::: ##:::: ##:::. ##:'####: ########::. ######::
..:::::..::........:::::..:::::..:::::..::....::........::::......:::
*/

// Mirrors compute_metrics() of the Python METIS partitioner so that both
// backends emit exactly the same fields.
//
// Convention notes (kept identical to the Python version):
//   * num_edges / edge_cut / edge_counts are counted over *directed arcs*
//     as stored in the CSR structure. For an undirected graph stored with
//     both directions, every cut edge is therefore counted twice - just as
//     in the DGL/METIS script, so the ratios stay comparable.
//   * edge_counts[p] counts the arcs whose *source* vertex lies in part p.
//   * node_balance / edge_balance are max/ideal ratios (1.0 == perfect).
struct partition_metrics_t {
    long num_nodes;
    long num_edges;
    vector<long> node_counts;
    vector<long> edge_counts;
    long edge_cut;
    double edge_cut_ratio;
    double node_balance;
    double edge_balance;
    double core_time;
};

partition_metrics_t compute_metrics(pulp_graph_t &g, int num_parts,
                                    int *parts) {
    partition_metrics_t metrics;
    metrics.num_nodes = (long)g.n;
    metrics.num_edges = (long)g.m;
    metrics.node_counts.assign(num_parts, 0);
    metrics.edge_counts.assign(num_parts, 0);
    metrics.edge_cut = 0;
    metrics.core_time = 0.0;

    long edge_cut = 0;
    vector<long> node_counts(num_parts, 0);
    vector<long> edge_counts(num_parts, 0);

    for (int v = 0; v < g.n; ++v) {
        int part_v = parts[v];
        if (part_v < 0 || part_v >= num_parts) {
            fprintf(stderr,
                    "Warning: vertex %d has invalid part id %d, skipping in "
                    "metrics.\n",
                    v, part_v);
            continue;
        }

        ++node_counts[part_v];

        long begin = g.out_degree_list[v];
        long end = g.out_degree_list[v + 1];
        edge_counts[part_v] += (end - begin);

        for (long j = begin; j < end; ++j)
            if (parts[g.out_array[j]] != part_v)
                ++edge_cut;
    }

    metrics.node_counts = node_counts;
    metrics.edge_counts = edge_counts;
    metrics.edge_cut = edge_cut;

    double ideal_nodes =
        num_parts ? (double)metrics.num_nodes / num_parts : 1.0;
    double ideal_edges =
        num_parts ? (double)metrics.num_edges / num_parts : 1.0;

    long max_nodes = 0;
    long max_edges = 0;
    for (int p = 0; p < num_parts; ++p) {
        if (node_counts[p] > max_nodes)
            max_nodes = node_counts[p];
        if (edge_counts[p] > max_edges)
            max_edges = edge_counts[p];
    }

    metrics.node_balance = (ideal_nodes > 0.0) ? max_nodes / ideal_nodes : NAN;
    metrics.edge_balance = (ideal_edges > 0.0) ? max_edges / ideal_edges : NAN;
    metrics.edge_cut_ratio = (metrics.num_edges > 0)
                                 ? (double)edge_cut / (double)metrics.num_edges
                                 : NAN;

    return metrics;
}

void print_balance_summary(partition_metrics_t &metrics, int num_parts) {
    long total = metrics.num_nodes;
    double ideal = num_parts ? (double)total / num_parts : 1.0;

    printf("-- Node partition balance -------------------------------\n");
    for (int p = 0; p < num_parts; ++p) {
        long cnt = metrics.node_counts[p];
        double pct = total ? 100.0 * cnt / (double)total : 0.0;
        printf("  Part %3d : %9ld nodes (%5.1f %%) | %9ld arcs\n", p, cnt, pct,
               metrics.edge_counts[p]);
    }
    printf("  Ideal per part           : %.1f\n", ideal);
    printf("  Node balance (max/ideal) : %.6f\n", metrics.node_balance);
    printf("  Edge balance (max/ideal) : %.6f\n", metrics.edge_balance);
    printf("  Edge cut / ratio         : %ld / %.6f\n", metrics.edge_cut,
           metrics.edge_cut_ratio);
    printf("---------------------------------------------------------\n");
}

// json.dump() writes bare NaN for math.nan, so we do the same here to keep
// downstream parsers happy with both backends.
string json_double(double value) {
    if (isnan(value))
        return string("NaN");
    if (isinf(value))
        return string(value > 0 ? "Infinity" : "-Infinity");
    char buf[64];
    snprintf(buf, sizeof(buf), "%.9f", value);
    return string(buf);
}

string json_escape(const string &s) {
    string out;
    for (size_t i = 0; i < s.size(); ++i) {
        char ch = s[i];
        if (ch == '"' || ch == '\\') {
            out += '\\';
            out += ch;
        } else if (ch == '\n')
            out += "\\n";
        else if (ch == '\t')
            out += "\\t";
        else
            out += ch;
    }
    return out;
}

string json_long_array(const vector<long> &values) {
    string out = "[";
    for (size_t i = 0; i < values.size(); ++i) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%ld", values[i]);
        if (i)
            out += ", ";
        out += buf;
    }
    out += "]";
    return out;
}

void write_metrics_json(const char *metrics_file, partition_metrics_t &metrics,
                        const string &dataset, int num_parts,
                        const string &objtype, double vert_balance_constraint,
                        double edge_balance_constraint, bool do_edge_balance,
                        bool do_maxcut_balance, int pulp_seed) {
    FILE *fp = fopen(metrics_file, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error: could not open metrics file %s for writing.\n",
                metrics_file);
        return;
    }

    fprintf(fp, "{\n");
    fprintf(fp, "  \"algorithm\": \"pulp\",\n");
    fprintf(fp, "  \"dataset\": \"%s\",\n", json_escape(dataset).c_str());
    fprintf(fp, "  \"k\": %d,\n", num_parts);
    fprintf(fp, "  \"objtype\": \"%s\",\n", json_escape(objtype).c_str());
    fprintf(fp, "  \"partitioning_type\": \"node\",\n");
    fprintf(fp, "  \"num_parts\": %d,\n", num_parts);
    fprintf(fp, "  \"num_nodes\": %ld,\n", metrics.num_nodes);
    fprintf(fp, "  \"num_edges\": %ld,\n", metrics.num_edges);
    fprintf(fp, "  \"node_counts\": %s,\n",
            json_long_array(metrics.node_counts).c_str());
    fprintf(fp, "  \"edge_counts\": %s,\n",
            json_long_array(metrics.edge_counts).c_str());
    fprintf(fp, "  \"edge_cut\": %ld,\n", metrics.edge_cut);
    fprintf(fp, "  \"edge_cut_ratio\": %s,\n",
            json_double(metrics.edge_cut_ratio).c_str());
    fprintf(fp, "  \"node_balance\": %s,\n",
            json_double(metrics.node_balance).c_str());
    fprintf(fp, "  \"edge_balance\": %s,\n",
            json_double(metrics.edge_balance).c_str());
    fprintf(fp, "  \"core_time\": %s,\n",
            json_double(metrics.core_time).c_str());
    // PULP specific settings (extra keys, ignored by the shared consumers)
    fprintf(fp, "  \"vert_balance_constraint\": %s,\n",
            json_double(vert_balance_constraint).c_str());
    fprintf(
        fp, "  \"edge_balance_constraint\": %s,\n",
        json_double(do_edge_balance ? edge_balance_constraint : NAN).c_str());
    fprintf(fp, "  \"do_maxcut_balance\": %s,\n",
            do_maxcut_balance ? "true" : "false");
    fprintf(fp, "  \"seed\": %d\n", pulp_seed);
    fprintf(fp, "}\n");

    fclose(fp);
}

string basename_no_ext(const string &path) {
    size_t slash = path.find_last_of("/\\");
    string base = (slash == string::npos) ? path : path.substr(slash + 1);
    size_t dot = base.find_last_of('.');
    if (dot != string::npos && dot != 0)
        base = base.substr(0, dot);
    return base;
}

// Same default as the Python script: metrics.json in the directory of the
// parts file.
string default_metrics_path(const string &out_file) {
    size_t slash = out_file.find_last_of("/\\");
    if (slash == string::npos)
        return string("metrics.json");
    return out_file.substr(0, slash + 1) + "metrics.json";
}

/*
'##::::'##::::'###::::'####:'##::: ##:
 ###::'###:::'## ##:::. ##:: ###:: ##:
 ####'####::'##:. ##::: ##:: ####: ##:
 ## ### ##:'##:::. ##:: ##:: ## ## ##:
 ##. #: ##: #########:: ##:: ##. ####:
 ##:.:: ##: ##.... ##:: ##:: ##:. ###:
 ##:::: ##: ##:::: ##:'####: ##::. ##:
..:::::..::..:::::..::....::..::::..::
*/
int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    srand(time(0));

    double main_start = timer();

    if (argc < 3) {
        print_usage_full(argv);
        exit(0);
    }

    int n = 0;
    long m = 0;
    int *out_array = NULL;
    long *out_degree_list = NULL;
    int *vertex_weights = NULL;
    long vertex_weights_sum = 0;
    int *edge_weights;
    char *graph_name = strdup(argv[1]);
    char *num_parts_str = strdup(argv[2]);
    int num_parts = atoi(num_parts_str);
    int *parts;
    int num_partitions = 1;

    double vert_balance = 1.10;
    double edge_balance = 1.50;
    char parts_out[1024];
    parts_out[0] = '\0';
    char parts_in[1024];
    parts_in[0] = '\0';
    char metrics_out[1024];
    metrics_out[0] = '\0';
    char dataset_name[1024];
    dataset_name[0] = '\0';

    bool do_bfs_init = true;
    bool do_lp_init = false;
    bool do_edge_balance = false;
    bool do_maxcut_balance = false;
    bool eval_quality = false;
    int pulp_seed = rand();

    int c;
    while ((c = getopt(argc, argv, "v:e:i:o:cs:lm:qn:M:")) != -1) {
        switch (c) {
        case 'v':
            vert_balance = strtod(optarg, NULL);
            break;
        case 'e':
            edge_balance = strtod(optarg, NULL);
            do_edge_balance = true;
            break;
        case 'i':
            strcat(parts_in, optarg);
            break;
        case 'o':
            strcat(parts_out, optarg);
            break;
        case 'c':
            do_maxcut_balance = true;
            break;
        case 'm':
            num_partitions = atoi(optarg);
            break;
        case 's':
            pulp_seed = atoi(optarg);
            break;
        case 'l':
            do_lp_init = true;
            do_bfs_init = false;
            break;
        case 'q':
            eval_quality = true;
            break;
        case 'n':
            strcat(dataset_name, optarg);
            break;
        case 'M':
            strcat(metrics_out, optarg);
            break;
        case '?':
            if (optopt == 'v' || optopt == 'e' || optopt == 'i' ||
                optopt == 'o' || optopt == 'm' || optopt == 'n' ||
                optopt == 'M')
                fprintf(stderr, "Option -%c requires an argument.\n", optopt);
            else if (isprint(optopt))
                fprintf(stderr, "Unknown option `-%c'.\n", optopt);
            else
                fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
            print_usage_full(argv);
        default:
            abort();
        }
    }

    double elt = 0.0;
    double avg = 0.0;

    string dataset = (strlen(dataset_name) != 0)
                         ? string(dataset_name)
                         : basename_no_ext(string(graph_name));
    string objtype = do_maxcut_balance ? string("maxcut") : string("cut");

    printf("Reading in %s ... ", graph_name);
    elt = timer();
    read_graph(graph_name, n, m, out_array, out_degree_list, vertex_weights,
               edge_weights, vertex_weights_sum);
    printf("n=%d m=%ld vw=%p vw_sum=%ld\n", n, m, (void *)vertex_weights,
           vertex_weights_sum);
    pulp_graph_t g = {n,
                      m,
                      out_array,
                      out_degree_list,
                      vertex_weights,
                      edge_weights,
                      vertex_weights_sum};
    elt = timer() - elt;
    printf("... Done: %9.6lf\n", elt);

    parts = new int[g.n];
    for (int i = 0; i < num_partitions; ++i) {
        if (strlen(parts_in) != 0) {
            printf("Reading in parts file %s ... ", parts_in);
            elt = timer();
            do_lp_init = false;
            do_bfs_init = false;
            read_parts(parts_in, g.n, parts);
            elt = timer() - elt;
            printf("Done: %9.6lf\n", elt);
        } else if (do_bfs_init)
            for (int j = 0; j < g.n; ++j)
                parts[j] = -1;
        else
            for (int j = 0; j < g.n; ++j)
                parts[j] = rand() % num_parts;

        pulp_part_control_t ppc = {
            vert_balance,    edge_balance,      do_lp_init, do_bfs_init,
            do_edge_balance, do_maxcut_balance, false,      pulp_seed};

        printf("\nBeginning partitioning ... ");
        elt = timer();
        pulp_run(&g, &ppc, parts, num_parts);
        elt = timer() - elt;
        avg += elt;
        printf("Partitioning Time: %9.6lf\n\n", elt);

        char temp_out[1024];
        temp_out[0] = '\0';
        strcat(temp_out, parts_out);
        if (strlen(temp_out) == 0) {
            strcat(temp_out, graph_name);
            strcat(temp_out, ".parts.");
            strcat(temp_out, num_parts_str);
        }
        if (num_partitions > 1) {
            strcat(temp_out, ".");
            stringstream ss;
            ss << i;
            strcat(temp_out, ss.str().c_str());
        }
        printf("writing parts file %s ... ", temp_out);
        elt = timer();
        write_parts(temp_out, g.n, parts);
        elt = timer() - elt;
        printf("Done: %9.6lf\n", elt);

        if (eval_quality)
            evaluate_quality(g, num_parts, parts);

        // ---- metrics
        // --------------------------------------------------------- core_time
        // is measured from the start of main (read + partition + write),
        // matching the convention of the Python partitioners.
        double core_time = timer() - main_start;

        partition_metrics_t metrics = compute_metrics(g, num_parts, parts);
        metrics.core_time = core_time;

        if (eval_quality)
            print_balance_summary(metrics, num_parts);

        string temp_metrics = (strlen(metrics_out) != 0)
                                  ? string(metrics_out)
                                  : default_metrics_path(string(temp_out));
        if (num_partitions > 1) {
            stringstream ss;
            ss << "." << i;
            temp_metrics += ss.str();
        }

        printf("\n[done]  Partitioning completed successfully.\n");
        printf("CODEx_METRICS "
               "algorithm=pulp "
               "dataset=%s "
               "partitioning_type=node "
               "k=%d "
               "num_nodes=%ld "
               "num_edges=%ld "
               "core_time=%.9f "
               "edge_cut=%ld "
               "edge_cut_ratio=%.9f "
               "node_balance=%.9f "
               "edge_balance=%.9f\n",
               dataset.c_str(), num_parts, metrics.num_nodes, metrics.num_edges,
               core_time, metrics.edge_cut, metrics.edge_cut_ratio,
               metrics.node_balance, metrics.edge_balance);

        printf("writing metrics file %s ... ", temp_metrics.c_str());
        // Repository-wide convention: measure until the final metrics write.
        metrics.core_time = timer() - main_start;
        write_metrics_json(temp_metrics.c_str(), metrics, dataset, num_parts,
                           objtype, vert_balance, edge_balance, do_edge_balance,
                           do_maxcut_balance, pulp_seed);
        printf("Done\n");
    }

    if (num_partitions > 1)
        printf("\nAverage partitioning time: %9.6lf\n", avg / num_partitions);

    delete[] parts;
    delete[] out_array;
    delete[] out_degree_list;
    free(graph_name);
    free(num_parts_str);

    return 0;
}
