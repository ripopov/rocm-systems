// MIT License
//
// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/rocprofiler-sdk/thread_trace/shared_trace_queue.hpp"
#include "lib/rocprofiler-sdk/thread_trace/hsa_util.hpp"

#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"

#include <algorithm>
#include <cstdint>
#include <map>

namespace rocprofiler
{
namespace thread_trace
{
namespace
{
struct agent_queue_t
{
    // Max triple-buffer staging size requested by any context on this agent; the
    // shared queue is created with this size so every context's flips fit.
    uint64_t        max_triple_buffer_size = 0;
    att_queue_ptr_t queue                  = {};
};

struct shared_queue_state_t
{
    std::map<uint64_t, agent_queue_t> agents = {};  // keyed by hsa_agent_t.handle
};

using locked_state_t = common::Synchronized<shared_queue_state_t>;

locked_state_t&
get_state()
{
    static auto*& _state = common::static_object<locked_state_t>::construct();
    return *CHECK_NOTNULL(_state);
}
}  // namespace

void
register_shared_queue_size(hsa_agent_t agent, uint64_t triple_buffer_size)
{
    get_state().wlock([&](shared_queue_state_t& state) {
        auto& entry                  = state.agents[agent.handle];
        entry.max_triple_buffer_size = std::max(entry.max_triple_buffer_size, triple_buffer_size);
    });
}

att_queue_t*
acquire_shared_queue(const hsa::AgentCache& agent)
{
    return get_state().wlock([&](shared_queue_state_t& state) -> att_queue_t* {
        auto& entry = state.agents[agent.get_hsa_agent().handle];

        // Reuse the agent's queue; the pre-pass sizes its staging to the agent max
        // up front, so the first acquire allocates and the rest just reuse.
        if(entry.queue) return entry.queue.get();

        entry.queue = make_att_queue(agent, entry.max_triple_buffer_size);
        return entry.queue.get();
    });
}

void
free_shared_queues()
{
    // Clearing runs each att_queue_ptr_t deleter (att_queue_destroy), so the HSA
    // queues are torn down exactly once here rather than per context.
    get_state().wlock([](shared_queue_state_t& state) { state.agents.clear(); });
}

}  // namespace thread_trace
}  // namespace rocprofiler
