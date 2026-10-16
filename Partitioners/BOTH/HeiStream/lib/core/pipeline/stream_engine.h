/****************************************************************************
 * stream_engine.h
 *****************************************************************************/

#ifndef CORE_STREAM_ENGINE_H_
#define CORE_STREAM_ENGINE_H_

#include <chrono>
#include <string>

#include "definitions.h"
#include "partition/partition_config.h"


namespace core::pipeline {

void run_stream_engine(StreamMode mode, Config& config, const std::string& graph_filename,
                       std::chrono::steady_clock::time_point main_start);

} // namespace core::pipeline


#endif /* CORE_STREAM_ENGINE_H_ */
