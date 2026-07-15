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
//
// Stress test for the per-agent shared trace buffer (AIPROFSDK-102). Allocates
// hundreds of device thread-trace contexts with varying parameters, each sized to
// ~1GB. Because only one trace can be active per agent at a time, all contexts on
// an agent share a single buffer, so this fits in memory; with a per-context buffer
// (the old implementation) ~400 x 1GB would exhaust device memory at init. The
// contexts are then run one at a time and we assert that most of them capture data.
//
// undefine NDEBUG so asserts are implemented
#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "thread-trace-callbacks.hpp"

#include <atomic>
#include <cstdlib>
#include <mutex>
#include <set>
#include <vector>

namespace ATTTest
{
namespace ManyContexts
{
rocprofiler_client_id_t* client_id = nullptr;

// Heap-allocated so we control destruction order relative to rocprofiler shutdown.
struct ToolState
{
    rocprofiler_context_id_t              tracing_ctx{};
    std::vector<rocprofiler_context_id_t> contexts{};

    // Sequencing of the one-active-trace-per-agent rotation. Kernel-dispatch
    // callbacks fire concurrently, so all rotation state is guarded by seq_mut.
    std::mutex seq_mut{};
    size_t     active_index{0};   // next context to run
    bool       trace_active{false};
    uint64_t   start_dispatch{0};

    // Contexts (by index) that delivered shader data.
    std::mutex        mut{};
    std::set<size_t>  captured{};
};

ToolState* state = nullptr;

// Number of contexts to allocate. Overridable so CI can tune memory/time.
size_t
num_contexts()
{
    static const size_t n =
        std::getenv("ATT_NUM_CONTEXTS") ? std::strtoul(std::getenv("ATT_NUM_CONTEXTS"), nullptr, 10)
                                        : 400;
    return n;
}

// Keep each context active across a few dispatches so it reliably captures one.
constexpr uint64_t CAPTURE_WINDOW = 2;

// Repro mode for the AILIKFD-39 teardown hang: start one context and never stop it,
// leaving SQTT active at process exit so rocprofiler teardown (resource_deinit ->
// agents.clear()) destroys still-active per-agent tracers one-by-one.
bool
leave_active()
{
    static const bool v = std::getenv("ATT_LEAVE_ACTIVE") != nullptr;
    return v;
}

void
shader_data_callback(rocprofiler_agent_id_t /* agent */,
                     int64_t /* se_id */,
                     void* /* se_data */,
                     size_t                                       data_size,
                     rocprofiler_thread_trace_shader_data_flags_t /* flags */,
                     rocprofiler_user_data_t                      userdata)
{
    if(data_size == 0 || state == nullptr) return;
    // userdata carries the context index this service was configured with.
    auto idx = reinterpret_cast<size_t>(userdata.ptr);

    std::unique_lock<std::mutex> lk(state->mut);
    state->captured.insert(idx);
}

// Drives the rotation: start the next context on a dispatch, stop it a couple of
// dispatches later, then advance. Only one context is ever active at a time.
void
dispatch_tracing_callback(rocprofiler_callback_tracing_record_t record,
                          rocprofiler_user_data_t* /* user_data */,
                          void* /* userdata */)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH) return;
    if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT) return;
    if(state == nullptr) return;

    assert(record.payload);
    auto* rdata = static_cast<rocprofiler_callback_tracing_kernel_dispatch_data_t*>(record.payload);
    auto  dispatch_id = rdata->dispatch_info.dispatch_id;

    // Serialize the whole check-then-act so concurrent dispatch callbacks can't
    // double-start or double-stop a context.
    std::unique_lock<std::mutex> lk(state->seq_mut);

    if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        if(!state->trace_active && state->active_index < state->contexts.size())
        {
            // Skip past any context that refuses to start rather than aborting.
            while(state->active_index < state->contexts.size() &&
                  rocprofiler_start_context(state->contexts[state->active_index]) !=
                      ROCPROFILER_STATUS_SUCCESS)
                ++state->active_index;

            if(state->active_index < state->contexts.size())
            {
                state->trace_active   = true;
                state->start_dispatch = dispatch_id;
            }
        }
        return;
    }

    // Completion phase: stop the active context once its capture window has elapsed.
    // In leave-active repro mode we never stop, so the first started context is still
    // active at process exit (AILIKFD-39 teardown race).
    assert(record.phase == ROCPROFILER_CALLBACK_PHASE_NONE);
    if(!leave_active() && state->trace_active &&
       dispatch_id >= state->start_dispatch + CAPTURE_WINDOW)
    {
        rocprofiler_stop_context(state->contexts[state->active_index]);
        state->trace_active = false;
        ++state->active_index;
    }
}

std::vector<rocprofiler_agent_id_t>
get_gpu_agents()
{
    std::vector<rocprofiler_agent_id_t> agents{};
    ROCPROFILER_CALL(
        rocprofiler_query_available_agents(
            ROCPROFILER_AGENT_INFO_VERSION_0,
            [](rocprofiler_agent_version_t, const void** _agents, size_t _num, void* _data) {
                auto* out = static_cast<std::vector<rocprofiler_agent_id_t>*>(_data);
                for(size_t i = 0; i < _num; ++i)
                {
                    const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(_agents[i]);
                    if(agent->type == ROCPROFILER_AGENT_TYPE_GPU) out->emplace_back(agent->id);
                }
                return ROCPROFILER_STATUS_SUCCESS;
            },
            sizeof(rocprofiler_agent_v0_t),
            &agents),
        "query agents");
    return agents;
}

int
tool_init(rocprofiler_client_finalize_t /* fini_func */, void* /* tool_data */)
{
    state = new ToolState{};

    auto agents = get_gpu_agents();
    if(agents.empty()) return 0;

    // Vary the configuration across contexts; buffer size is intentionally large so
    // the per-agent shared buffer (sized to the max) is ~1GB.
    constexpr uint64_t GB               = 1ull << 30;
    const uint64_t     buffer_sizes[]   = {1 * GB, 512ull << 20};
    const uint32_t     simd_selects[]   = {0xF, 0x1, 0x2, 0x4, 0x8};

    const size_t N = num_contexts();
    state->contexts.reserve(N);

    // Sequence trace start/stop over kernel dispatches.
    ROCPROFILER_CALL(rocprofiler_create_context(&state->tracing_ctx), "context creation");
    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(state->tracing_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                                                       nullptr,
                                                       0,
                                                       dispatch_tracing_callback,
                                                       nullptr),
        "dispatch tracing service configure");

    for(size_t i = 0; i < N; ++i)
    {
        rocprofiler_context_id_t ctx{};
        ROCPROFILER_CALL(rocprofiler_create_context(&ctx), "context creation");

        uint32_t target_cu = static_cast<uint32_t>(i % 4);          // CUs in [0,3]
        uint32_t se_mask    = (i % 2) != 0 ? 0x3u : 0x1u;
        uint32_t simd       = simd_selects[i % (sizeof(simd_selects) / sizeof(uint32_t))];
        uint64_t buf_size   = buffer_sizes[i % 2];

        auto params = std::vector<rocprofiler_thread_trace_parameter_t>{};
        params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_TARGET_CU, {target_cu}});
        params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_SIMD_SELECT, {simd}});
        params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK, {se_mask}});
        params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_BUFFER_SIZE, {buf_size}});

        rocprofiler_user_data_t user{};
        user.ptr = reinterpret_cast<void*>(i);

        bool ok = true;
        for(auto agent : agents)
        {
            auto status = rocprofiler_configure_device_thread_trace_service(
                ctx, agent, params.data(), params.size(), shader_data_callback, user);
            if(status != ROCPROFILER_STATUS_SUCCESS) ok = false;
        }

        int valid = 0;
        rocprofiler_context_is_valid(ctx, &valid);
        if(ok && valid != 0) state->contexts.emplace_back(ctx);
    }

    std::cerr << "[many-contexts] configured " << state->contexts.size() << " / " << N
              << " contexts" << std::endl;

    ROCPROFILER_CALL(rocprofiler_start_context(state->tracing_ctx), "tracing context start");
    return 0;
}

void
tool_fini(void* /* tool_data */)
{
    if(state == nullptr) return;

    if(leave_active())
    {
        // Deliberately do NOT stop the trace context: leave SQTT active so rocprofiler
        // teardown destroys the still-active per-agent tracers one-by-one (AILIKFD-39).
        rocprofiler_stop_context(state->tracing_ctx);
        std::cerr << "[many-contexts] leave-active: trace left running into teardown"
                  << std::endl;
        return;  // keep `state` alive; process is exiting
    }

    size_t active_index = 0;
    {
        std::unique_lock<std::mutex> lk(state->seq_mut);
        // Make sure nothing is left active before shutdown.
        if(state->trace_active && state->active_index < state->contexts.size())
            rocprofiler_stop_context(state->contexts[state->active_index]);
        active_index = state->active_index;
    }
    rocprofiler_stop_context(state->tracing_ctx);

    size_t configured = state->contexts.size();
    size_t ran        = std::min(active_index, configured);
    size_t captured   = 0;
    {
        std::unique_lock<std::mutex> lk(state->mut);
        captured = state->captured.size();
    }

    std::cerr << "[many-contexts] configured=" << configured << " ran=" << ran
              << " captured=" << captured << std::endl;

    // Every context must have configured and allocated its (shared) buffer, which is
    // the memory-sharing proof: this would OOM with a per-context buffer.
    assert(configured == num_contexts() && "not all contexts configured/allocated");

    // Every context should have had a turn.
    assert(ran == configured && "not every context was run (need more dispatches)");

    // Not every target_cu is guaranteed to have active waves, so require that MOST
    // configs produced data rather than all.
    assert(captured * 4 >= configured * 3 && "most configs should capture data");

    delete state;
    state = nullptr;
}

}  // namespace ManyContexts
}  // namespace ATTTest

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t /* version */,
                      const char* /* runtime_version */,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    if(priority > 0) return nullptr;

    id->name                             = "ATT_test_many_contexts";
    ATTTest::ManyContexts::client_id     = id;

    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &ATTTest::ManyContexts::tool_init,
                                            &ATTTest::ManyContexts::tool_fini,
                                            nullptr};
    return &cfg;
}
