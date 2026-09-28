// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/callback_dispatcher.h"

#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace rocjitsu {
namespace {

TEST(CallbackDispatcherTest, OwnerCallsAreDirectAndReentrant) {
  CallbackDispatcher dispatcher;
  EXPECT_EQ(dispatcher.invoke([&] { return dispatcher.invoke([] { return 42; }); }), 42);
  pollfd descriptor{dispatcher.fd(), POLLIN, 0};
  EXPECT_EQ(::poll(&descriptor, 1, 0), 0);
}

TEST(CallbackDispatcherTest, WorkerCallsWakeOwnerAndReturnValues) {
  CallbackDispatcher dispatcher;
  const auto owner = std::this_thread::get_id();
  auto result = std::async(std::launch::async, [&] {
    return dispatcher.invoke([&, value = std::make_unique<int>(42)]() mutable {
      EXPECT_EQ(std::this_thread::get_id(), owner);
      return std::move(value);
    });
  });
  pollfd descriptor{dispatcher.fd(), POLLIN, 0};
  ASSERT_EQ(::poll(&descriptor, 1, 5000), 1);
  dispatcher.dispatch();
  EXPECT_EQ(*result.get(), 42);
  EXPECT_EQ(::poll(&descriptor, 1, 0), 0);
}

TEST(CallbackDispatcherTest, CallbackExceptionsReturnToWorker) {
  CallbackDispatcher dispatcher;
  dispatcher.run_while_dispatching([&] {
    EXPECT_THROW(dispatcher.invoke([] { throw std::runtime_error("callback failed"); }),
                 std::runtime_error);
    EXPECT_EQ(dispatcher.invoke([] { return 7; }), 7);
  });
}

TEST(CallbackDispatcherTest, ShutdownPumpsCallbacksFromAllWorkersBeforeJoining) {
  CallbackDispatcher dispatcher;
  const auto owner = std::this_thread::get_id();
  unsigned calls = 0;
  dispatcher.run_while_dispatching([&] {
    std::vector<std::jthread> workers;
    for (unsigned worker = 0; worker < 8; ++worker) {
      workers.emplace_back([&] {
        for (unsigned call = 0; call < 32; ++call) {
          dispatcher.invoke([&] {
            EXPECT_EQ(std::this_thread::get_id(), owner);
            ++calls;
          });
        }
      });
    }
  });
  EXPECT_EQ(calls, 256u);
  pollfd descriptor{dispatcher.fd(), POLLIN, 0};
  EXPECT_EQ(::poll(&descriptor, 1, 0), 0);
}

TEST(CallbackDispatcherTest, ShutdownFailureIsReportedAfterCleanup) {
  CallbackDispatcher dispatcher;
  bool cleaned = false;
  EXPECT_THROW(dispatcher.run_while_dispatching([&] {
    dispatcher.invoke([&] { cleaned = true; });
    throw std::runtime_error("stop failed");
  }),
               std::runtime_error);
  EXPECT_TRUE(cleaned);
  dispatcher.run_while_dispatching([] {});
}

TEST(CallbackDispatcherTest, OnlyOwnerCanDispatchOrDriveShutdown) {
  CallbackDispatcher dispatcher;
  std::jthread worker([&] {
    EXPECT_THROW(dispatcher.dispatch(), std::logic_error);
    EXPECT_THROW(dispatcher.run_while_dispatching([] {}), std::logic_error);
  });
}

} // namespace
} // namespace rocjitsu
