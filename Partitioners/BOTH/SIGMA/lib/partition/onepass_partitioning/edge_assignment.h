/******************************************************************************
 * edge_assignment.h
 *
 * Memory-efficient replacement for std::vector<std::bitset<MAX_NUM_PARTITION>>.
 * Stores per-node partition membership as a flat array of uint64_t words,
 * sized to ceil(k/64) words per node instead of ceil(1024/64)=16 words.
 * For k=128 this reduces memory from 128 bytes/node to 16 bytes/node (8x).
 *****************************************************************************/

#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

#include "definitions.h"

class EdgeAssignmentArray {
public:
    // ------------------------------------------------------------------
    // BitProxy: reference to a single bit, supporting read and assignment
    // ------------------------------------------------------------------
    struct BitProxy {
        uint64_t* word;
        uint64_t  mask;

        BitProxy& operator=(bool val) noexcept {
            if (val) *word |=  mask;
            else     *word &= ~mask;
            return *this;
        }
        operator bool() const noexcept { return (*word & mask) != 0; }
    };

    // ------------------------------------------------------------------
    // NodeProxy: view of one node's bitset row (ceil(k/64) words)
    // ------------------------------------------------------------------
    struct NodeProxy {
        uint64_t*   row;
        std::size_t stride;  // number of uint64_t words in this row

        // Non-const: returns BitProxy for set/read via operator=
        BitProxy operator[](PartitionID p) noexcept {
            return BitProxy{ row + (p / 64), uint64_t(1) << (p % 64) };
        }
        // Const: direct bool read
        bool operator[](PartitionID p) const noexcept {
            return (row[p / 64] >> (p % 64)) & 1;
        }

        std::size_t count() const noexcept {
            std::size_t c = 0;
            for (std::size_t w = 0; w < stride; ++w)
                c += __builtin_popcountll(row[w]);
            return c;
        }

        void reset() noexcept {
            std::memset(row, 0, stride * sizeof(uint64_t));
        }
    };

    // ------------------------------------------------------------------
    // Iterator over all node rows (for range-based for and std::for_each)
    // ------------------------------------------------------------------
    struct Iterator {
        EdgeAssignmentArray* arr;
        std::size_t          idx;

        NodeProxy  operator*()  const noexcept { return (*arr)[idx]; }
        Iterator&  operator++() noexcept { ++idx; return *this; }
        bool operator!=(const Iterator& o) const noexcept { return idx != o.idx; }
    };

    // ------------------------------------------------------------------
    // Constructor: allocates n_nodes rows of ceil(k/64) zero-initialised
    // uint64_t words. k must be > 0.
    // ------------------------------------------------------------------
    EdgeAssignmentArray(std::size_t n_nodes, PartitionID k)
        : n_nodes_(n_nodes),
          stride_((static_cast<std::size_t>(k) + 63) / 64),
          data_(n_nodes * ((static_cast<std::size_t>(k) + 63) / 64), 0)
    {}

    // ------------------------------------------------------------------
    // Element access - returns NodeProxy for the given node
    // ------------------------------------------------------------------
    NodeProxy operator[](std::size_t node_id) noexcept {
        return NodeProxy{ data_.data() + node_id * stride_, stride_ };
    }
    NodeProxy operator[](std::size_t node_id) const noexcept {
        return NodeProxy{ const_cast<uint64_t*>(data_.data() + node_id * stride_), stride_ };
    }

    // ------------------------------------------------------------------
    // Iteration (begin/end for range-based for and std::for_each)
    // ------------------------------------------------------------------
    Iterator begin() noexcept { return {this, 0}; }
    Iterator end()   noexcept { return {this, n_nodes_}; }

    std::size_t size() const noexcept { return n_nodes_; }

    // Raw read-only pointer to a node's bitset row (for precomputation hot-paths)
    const uint64_t* row_ptr(std::size_t node_id) const noexcept {
        return data_.data() + node_id * stride_;
    }
    // Number of uint64_t words per node row (ceil(k/64))
    std::size_t stride() const noexcept { return stride_; }

private:
    std::size_t           n_nodes_;
    std::size_t           stride_;   // ceil(k/64)
    std::vector<uint64_t> data_;     // flat storage: n_nodes_ * stride_ words
};
