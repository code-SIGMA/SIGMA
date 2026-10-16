#include <algorithm>
#include <random>
#include <filesystem>

#include "twophasepartitioner.hpp"

void sorted_com_prepartition_forwarder(void* object, std::vector<edge_t> edges);
void hdrf_forwarder(void* object, std::vector<edge_t> edges);
void linear_forwarder(void* object, std::vector<edge_t> edges);

DECLARE_bool(write_parts);
DECLARE_string(parts_filename);
DECLARE_string(prepartitioner_type);
DECLARE_bool(use_hdrf);
TwoPhasePartitioner::TwoPhasePartitioner(Globals& GLOBALS) : globals(GLOBALS)
{
    edge_load.resize(globals.NUM_PARTITIONS, 0);
    vertex_partition_matrix.resize(globals.NUM_VERTICES);

    partition_volume.resize(globals.NUM_PARTITIONS, 0);
    com2part.resize(globals.NUM_VERTICES + 1, 0);

    // since 2PS has a hard-bound for partition size and pre-partitioner allocates the certain part of the
    // edges based on "perfect partitioning logic", we don't need min_load to play a role on the partition score 
    // calculation but to be tie-breaker in case there are more than one the same score partitions. 
    if (!FLAGS_prepartitioner_type.empty()) 
    {
        tie_breaker_min_load = true;
        min_load = 1;
    }

}

void TwoPhasePartitioner::set_partitioning_time(double time)
{
    partitioning_time = time;
}

double TwoPhasePartitioner::get_partitioning_time() const
{
    return partitioning_time;
}

void TwoPhasePartitioner::do_sorted_com_prepartitioning(std::vector<edge_t> edges)
{
    DLOG(INFO) << "Processing " << edges.size() << " edges";
    for (auto& e : edges)
    {
        auto& com_u = communities[e.first];
        auto& com_v = communities[e.second];
        if ((com_u == com_v) || com2part[com_u] == com2part[com_v])
        {
            auto partition = com2part[com_u];
            auto load = edge_load[partition];
            if (load >= globals.MAX_PARTITION_LOAD) 
            {
                partition = find_max_score_partition(e);
            }

            update_vertex_partition_matrix(e, partition);            
            update_min_max_load(partition);

            if (FLAGS_write_parts){
                write_edge(e, partition);
            }
        }
    }
}

void TwoPhasePartitioner::write_edge(edge_t e, int p){
    partitioned_edges[p].emplace_back(e);
    partitioned_edges[p].emplace_back(e);
    if (part_file.is_open()) {
        part_file << e.first << " " << e.second << " " << p << std::endl;
    }
}

void TwoPhasePartitioner::write_partitioned_edges_by_pid_binary(const std::string& output_path) {
    DLOG(INFO) << "Starting to write partitioned edges in binary format...";
    DLOG(INFO) << "Output path: " << output_path;
    DLOG(INFO) << "Total partitions in map: " << partitioned_edges.size();

    for (const auto& [pid, edges]: partitioned_edges) {
        DLOG(INFO) << "Processing partition " << pid << " with " << edges.size() << " edges";

        std::ostringstream oss;
        oss << output_path << "/part" << pid << "/partition_" << pid << ".bin";
        std::string filename = oss.str();

        std::filesystem::path dir =
            std::filesystem::path(filename).parent_path();
        if (!std::filesystem::exists(dir)) {
            try {
                std::filesystem::create_directories(dir);
                DLOG(INFO) << "Created directories: " << dir
                          << std::endl;
            } catch (const std::exception &e) {
                LOG(ERROR) << "Could not create directories: "
                          << e.what() << std::endl;
                continue;
            }
        }

        DLOG(INFO) << "Writing to binary file: " << filename;

        std::ofstream out(filename, std::ios::binary);
        if(!out.is_open()) {
            LOG(ERROR) << "Could not open binary file: " << filename;
            continue;
        }

        // Write the number of edges first (optional, for verification)
        std::uint64_t edge_count = edges.size();
        out.write(reinterpret_cast<const char*>(&edge_count), sizeof(edge_count));

        // Write all edges in binary format
        for (const auto& e: edges) {
            out.write(reinterpret_cast<const char*>(&e.first), sizeof(e.first));
            out.write(reinterpret_cast<const char*>(&e.second), sizeof(e.second));
        }
        out.close();

        DLOG(INFO) << "Successfully wrote " << edges.size() << " edges to " << filename;

    }

    DLOG(INFO) << "Finished writing partitioned binary edges.";
}

void TwoPhasePartitioner::export_dgl_partition_metadata(const std::string& output_path, const std::string& graph_name) {
    std::string path = output_path + "/" + graph_name + ".json";
    std::ofstream meta_file(path);
    meta_file << "{\n";
    meta_file << "  \"graph_name\": \"" << graph_name << "\",\n";
    meta_file << "  \"num_nodes\": " << globals.NUM_VERTICES << ",\n";
    meta_file << "  \"num_edges\": " << globals.NUM_EDGES << ",\n";
    meta_file << "  \"part_method\": \"TwoPhase\",\n";
    meta_file << "  \"num_parts\": " << globals.NUM_PARTITIONS << ",\n";
    meta_file << "  \"halo_hops\": 0,\n";

    // node_map
    meta_file << "  \"node_map\": [";
    for (size_t i = 0; i < vertex_partition_matrix.size(); ++i) {
        for (size_t p = 0; p < globals.NUM_PARTITIONS; ++p) {
            if (vertex_partition_matrix[i][p]) {
                meta_file << p;
                break;
            }
        }
        if (i + 1 < vertex_partition_matrix.size()) meta_file << ", ";
    }

    meta_file << "], \n";

    meta_file << "  \"edge_map\": 0";

    // parts
    for (size_t p = 0; p < globals.NUM_PARTITIONS; ++p) {
        meta_file << ",\n  \"part-" << p << "\": {\n";
        meta_file << "    \"node_feats\": \"part" << p << "/node_feat.dgl\",\n";
        meta_file << "    \"edge_feats\": \"part" << p << "/edge_feat.dgl\",\n";
        meta_file << "    \"part_graph\": \"part" << p << "/graph.dgl\"\n";
        meta_file << "  }";
    }

    meta_file << "\n}\n";
    meta_file.close();
}

void TwoPhasePartitioner::export_node_lists(const std::string& output_path) {
    LOG(INFO) << "export node lists";
    for (size_t p = 0; p < globals.NUM_PARTITIONS; ++p) {
        const std::string path = output_path + "/part" + std::to_string(p);
        std::filesystem::create_directories(path);
        std::ofstream ofs(path + "/nodes.txt");
        for(size_t v = 0; v < vertex_partition_matrix.size(); ++v) {
            if(vertex_partition_matrix[v][p]) {
                ofs << v << "\n";
            }
        }
        ofs.close();
    }
}

void TwoPhasePartitioner::perform_prepartition_and_partition(std::vector<uint32_t> coms, std::vector<uint64_t> vols, std::vector<double> qscores)
{
    Timer timer;
    timer.start();
    
    communities = std::move(coms);
    volumes = std::move(vols);
    quality_scores = std::move(qscores);

    // emplace each community with its volume size
    std::vector<std::pair<uint64_t, uint32_t>> sorted_communities;
    for (size_t i = 0; i < volumes.size(); ++i)
    {
        sorted_communities.emplace_back(volumes[i], i);
    }
    std::sort(sorted_communities.rbegin(), sorted_communities.rend()); // sort in descending order

    for (auto& com_volume_pair : sorted_communities)
    {
        // the rest of the communities has volume 0
        if (com_volume_pair.first == 0) break; 

        int min_p = find_min_vol_partition();
        partition_volume[min_p] += com_volume_pair.first;
        // assign partition min_p to the current community
        com2part[com_volume_pair.second] = min_p; 
    }
    timer.stop();
    LOG(INFO) << "Runtime for assigning commmunities to partitions " <<" [sec]: " << timer.get_time(); 

    LOG(INFO) << "Writing out partitions is enabled? " << FLAGS_write_parts << std::endl;
    if (FLAGS_write_parts){
        part_file.open(FLAGS_parts_filename + ".txt");
    }

    globals.read_and_do(sorted_com_prepartition_forwarder, this, "sorted communities prepartitioning");

    uint64_t sum_edge_load = 0;
    for (size_t i = 0; i < edge_load.size(); i++) sum_edge_load += edge_load[i];
    
    LOG(INFO) << "The number of edges prepartitioned: " << sum_edge_load;
    
    if (FLAGS_use_hdrf){
    	globals.read_and_do(hdrf_forwarder, this, "partitions with hdrf streaming edge partitioning");
    }
    else{
    	globals.read_and_do(linear_forwarder, this, "partitions with linear streaming edge partitioning");
    }

    if (FLAGS_write_parts){
        part_file.close();
    }
}

void TwoPhasePartitioner::perform_partitioning()
{
    LOG(INFO) << "Writing out partitions is enabled? " << FLAGS_write_parts << std::endl;
    if (FLAGS_write_parts){
        part_file.open(FLAGS_parts_filename + ".txt");
    }
    // start partitioner with hdrf
    globals.read_and_do(hdrf_forwarder, this, "partitions with hdrf");
    if (FLAGS_write_parts){
        part_file.close();
    }
}

void TwoPhasePartitioner::do_hdrf(std::vector<edge_t> &edges){
	for (auto &e : edges)
	    {
	        // TwoPhasePartitioner relies on partitional degrees, while pre-partitioner calculates the degrees in advance.
	        // If pre-partitioner is not performed, calculate degrees;
	        if (FLAGS_prepartitioner_type.empty())
	        {
	            ++globals.DEGREES[e.first];
	            ++globals.DEGREES[e.second];
	        }
	        else
	        {
	            auto &com_u = communities[e.first];
	            auto &com_v = communities[e.second];
	            if (com_u == com_v || com2part[com_u] == com2part[com_v])
	            {
	                continue;
	            }
	        }

	        int max_p = find_max_score_partition_hdrf(e);
	        update_vertex_partition_matrix(e, max_p);
	        update_min_max_load(max_p);
	        if (FLAGS_write_parts){
	            write_edge(e, max_p);
	        }
	    }
}


int TwoPhasePartitioner::find_max_score_partition_hdrf(edge_t& e)
{
	auto degree_u = globals.DEGREES[e.first];
	    auto degree_v = globals.DEGREES[e.second];

	    uint32_t sum;
	    double max_score = 0;
	    uint32_t max_p = 0;
	    double bal, gv, gu;

	    for (int p=0; p<globals.NUM_PARTITIONS; p++)
	    {
	        if (!FLAGS_prepartitioner_type.empty() && edge_load[p] >= globals.MAX_PARTITION_LOAD)
	        {
	            continue;
	        }

	        gu = 0, gv = 0;
	        sum = degree_u + degree_v;
	        if (vertex_partition_matrix[e.first][p])
	        {
	            gu = degree_u;
	            gu/=sum;
	            gu = 1 + (1-gu);
	        }
	        if (vertex_partition_matrix[e.second][p])
	        {
	            gv = degree_v;
	            gv /= sum;
	            gv = 1 + (1-gv);
	        }

	        bal = max_load - edge_load[p];
	        if (min_load != UINT64_MAX) bal /= epsilon + max_load - min_load;
	        double score_p = gu + gv + globals.LAMBDA*bal;
	        if (score_p < 0)
	        {
	            LOG(ERROR) << "ERROR: score_p < 0";
	            LOG(ERROR) << "gu: " << gu;
	            LOG(ERROR) << "gv: " << gv;
	            LOG(ERROR) << "bal: " << bal;
	            exit(-1);
	        }
	        if (score_p > max_score)
	        {
	            max_score = score_p;
	            max_p = p;
	        }
	    }
	    return max_p;

}

void TwoPhasePartitioner::do_linear(std::vector<edge_t> &edges)
{
    for (auto &e : edges)
    {
        // TwoPhasePartitioner relies on partitional degrees, while pre-partitioner calculates the degrees in advance.
        // If pre-partitioner is not performed, calculate degrees;
        if (FLAGS_prepartitioner_type.empty())
        {
            ++globals.DEGREES[e.first];
            ++globals.DEGREES[e.second];
        }
        else
        {
            auto &com_u = communities[e.first];
            auto &com_v = communities[e.second];
            if (com_u == com_v || com2part[com_u] == com2part[com_v])
            {
                continue;
            }
        }
        
        int max_p = find_max_score_partition(e);
        update_vertex_partition_matrix(e, max_p);
        update_min_max_load(max_p);
        if (FLAGS_write_parts){
            write_edge(e, max_p);
        }
    }
}

int TwoPhasePartitioner::find_max_score_partition(edge_t& e)
{
    auto degree_u = globals.DEGREES[e.first];
    auto degree_v = globals.DEGREES[e.second];
    uint32_t com_part_u = com2part[communities[e.first]];
    uint32_t com_part_v = com2part[communities[e.second]];
    


    uint64_t sum, sum_of_volumes;
    uint64_t external_degrees_u, external_degrees_v;
    double max_score = 0;
    uint32_t max_p = 0;
    double bal, gv, gu, gv_c, gu_c;

    // if target partitions based on vertex clusters are already full
    if (edge_load[com_part_u] >= globals.MAX_PARTITION_LOAD || edge_load[com_part_v] >= globals.MAX_PARTITION_LOAD){
    	if (degree_u > degree_v){
    		max_p = e.first % globals.NUM_PARTITIONS;
    	}
    	else {
    		max_p = e.second % globals.NUM_PARTITIONS;
    	}
//    	max_p = ps[pos];
    	if (edge_load[max_p] >= globals.MAX_PARTITION_LOAD){
    		// assign to min loaded partition
    		uint64_t min_load = -1;
    		uint32_t min_p = 0;
    		for (uint32_t i = 0; i < globals.NUM_PARTITIONS; i++){
    			if (edge_load[i] < min_load){
    				min_load = edge_load[i];
    				min_p = i;
    			}
    		}
    		max_p = min_p;
    	}

    	return max_p;
    }
    else{
    	std::vector<uint32_t> ps{com_part_u, com_part_v};

    	for (uint32_t p : ps)
    	{
			if (!FLAGS_prepartitioner_type.empty() && edge_load[p] >= globals.MAX_PARTITION_LOAD)
			{
				continue;
			}

			gu = 0, gv = 0, gu_c = 0, gv_c = 0;
			sum = degree_u + degree_v;
			sum_of_volumes = volumes[communities[e.first]] + volumes[communities[e.second]];

			if (vertex_partition_matrix[e.first][p])
			{
				gu = degree_u;
				gu /= sum;
				gu = 1 + (1-gu);
				if (com2part[communities[e.first]] == p){
	//            	size_t denominator_u = std::min(volumes[communities[e.first]], 2*globals.NUM_EDGES - volumes[communities[e.first]]);
	//            	external_degrees_u = quality_scores[communities[e.first]] * denominator_u;
					gu_c = volumes[communities[e.first]];
					gu_c /= sum_of_volumes;

	//                gu_c += (1 - quality_scores[communities[e.first]]);
		//            gu_c = 1 + gu_c;
				}

			}
			if (vertex_partition_matrix[e.second][p])
			{
				gv = degree_v;
				gv /= sum;
				gv = 1 + (1-gv);
				if (com2part[communities[e.second]] == p){
	//                size_t denominator_v = std::min(volumes[communities[e.second]], 2*globals.NUM_EDGES - volumes[communities[e.second]]);
	//                external_degrees_v = quality_scores[communities[e.second]] * denominator_v;
					gv_c = volumes[communities[e.second]];
					gv_c /= sum_of_volumes;
	//                gv_c += (1 - quality_scores[communities[e.second]]);
		//        	gv_c = 1 + gv_c;
				}
			}

			double score_p = gu + gv +  gu_c +  gv_c;
			if (score_p < 0)
			{
				LOG(ERROR) << "ERROR: score_p < 0";
				LOG(ERROR) << "gu: " << gu;
				LOG(ERROR) << "gv: " << gv;
				LOG(ERROR) << "gu_c: " << gu_c;
				LOG(ERROR) << "gv_c: " << gv_c;
				LOG(ERROR) << "quality scores u: " << quality_scores[communities[e.first]];
				LOG(ERROR) << "quality scores v: " << quality_scores[communities[e.second]];
				LOG(ERROR) << "bal: " << bal;
				exit(-1);
			}
			if (score_p >= max_score)
			{
				max_score = score_p;
				max_p = p;
			}
		}
    }
    return max_p;
}

void TwoPhasePartitioner::update_vertex_partition_matrix(edge_t& e, int max_p)
{
    if (max_p >= MAX_NUM_PARTITION) {
        LOG(FATAL) << "max_p (" << max_p << ") exceeds MAX_NUM_PARTITION (" << MAX_NUM_PARTITION << ")";
        exit(1);
    }
    vertex_partition_matrix[e.first][max_p] = true;
    vertex_partition_matrix[e.second][max_p] = true;
}

void TwoPhasePartitioner::update_min_max_load(int max_p)
{
    auto& load = ++edge_load[max_p];
    if (load > max_load) max_load = load; 

//    if (!tie_breaker_min_load)
//    {
//        min_load = edge_load[0];
//        for (int i = 1; i < globals.NUM_PARTITIONS; i++)
//        {
//            if (edge_load[i] < min_load)
//            {
//                min_load = edge_load[i];
//            }
//        }
//    }
}

int TwoPhasePartitioner::find_min_vol_partition()
{
    auto min_p_volume = partition_volume[0];
    int min_p = 0;
    for (int i = 1; i < globals.NUM_PARTITIONS; i++)
    {
        if (partition_volume[i] < min_p_volume)
        {
            min_p_volume = partition_volume[i];
            min_p = i;
        }
    }
    return min_p;    
}

std::vector<uint64_t>& TwoPhasePartitioner::get_edge_load()
{
    return edge_load;
}

std::vector<std::bitset<MAX_NUM_PARTITION>> TwoPhasePartitioner::get_vertex_partition_matrix()
{
    return vertex_partition_matrix;
}

const std::unordered_map<int, std::vector<edge_t>>& TwoPhasePartitioner::get_partitioned_edges() const {
    DLOG(INFO) << "Partitioned edge counts: " << partitioned_edges.size();
    return partitioned_edges;
}


void sorted_com_prepartition_forwarder(void* object, std::vector<edge_t> edges)
{
    static_cast<TwoPhasePartitioner*>(object)->do_sorted_com_prepartitioning(edges);
}

void linear_forwarder(void* object, std::vector<edge_t> edges)
{
    static_cast<TwoPhasePartitioner*>(object)->do_linear(edges);
}

void hdrf_forwarder(void* object, std::vector<edge_t> edges)
{
    static_cast<TwoPhasePartitioner*>(object)->do_hdrf(edges);
}
