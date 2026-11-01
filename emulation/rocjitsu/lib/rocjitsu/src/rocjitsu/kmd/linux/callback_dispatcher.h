// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_KMD_LINUX_CALLBACK_DISPATCHER_H_
#define ROCJITSU_KMD_LINUX_CALLBACK_DISPATCHER_H_

#include "util/unique_handle.h"

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <functional>
#include <future>
#include <list>
#include <memory>
#include <mutex>
#include <poll.h>
#include <stdexcept>
#include <sys/eventfd.h>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>

namespace rocjitsu {

/// Dispatch embedded-device callbacks onto the thread that constructed this
/// object. RPC workers may hold KFD locks while invoking callbacks: the owner
/// must not acquire those locks while servicing this queue. It polls fd() and
/// calls dispatch(); the descriptor is borrowed and must not be read or closed.
/// All producers must finish before destruction. This is callback scheduling,
/// not a second RPC transport or an executor for blocking KFD event waits.
class CallbackDispatcher {
public:
  CallbackDispatcher() : notification_(::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK)) {
    if (!notification_)
      throw std::system_error(errno, std::generic_category(), "callback eventfd");
  }

  int fd() const { return notification_.get(); }

  /// Wake the owner for transport state changes that need no device callback.
  /// Safe while holding transport locks; dispatch() also consumes these wakes.
  void wake() noexcept { notify(); }

  template <typename Function> auto invoke(Function &&function) -> std::invoke_result_t<Function> {
    if (std::this_thread::get_id() == owner_)
      return std::invoke(std::forward<Function>(function));
    using Result = std::invoke_result_t<Function>;
    auto task = std::make_shared<std::packaged_task<Result()>>(std::forward<Function>(function));
    auto result = task->get_future();
    {
      std::lock_guard lock(mutex_);
      callbacks_.push_back({{}, [task] { (*task)(); }});
      notify();
    }
    return result.get();
  }

  /// Wait on a worker until an owner-thread, nonthrowing readiness probe passes.
  /// Deferred tasks retain their promises and never block the dispatching owner.
  template <typename Ready, typename Function>
  auto invoke_when(Ready &&ready, Function &&function) -> std::invoke_result_t<Function> {
    if (std::this_thread::get_id() == owner_) {
      if (!std::invoke(ready))
        throw std::logic_error("owner cannot wait for its own callback readiness");
      return std::invoke(std::forward<Function>(function));
    }
    using Result = std::invoke_result_t<Function>;
    auto task = std::make_shared<std::packaged_task<Result()>>(std::forward<Function>(function));
    auto result = task->get_future();
    {
      std::lock_guard lock(mutex_);
      callbacks_.push_back({std::forward<Ready>(ready), [task] { (*task)(); }});
      notify();
    }
    return result.get();
  }

  /// Execute one batch without holding the scheduling lock. New requests
  /// submitted during a callback keep the fd readable for the next dispatch.
  void dispatch() {
    require_owner();
    // Splicing both the batch and deferred tasks allocates nothing: teardown
    // pumping must remain possible while a producer unwinds allocation failure.
    std::list<PendingCallback> batch;
    {
      std::lock_guard lock(mutex_);
      uint64_t value;
      while (::read(fd(), &value, sizeof(value)) < 0) {
        if (errno == EINTR)
          continue;
        if (errno != EAGAIN)
          std::terminate(); // Owned eventfd was closed/corrupted by the host.
        break;
      }
      batch.splice(batch.end(), callbacks_);
    }
    while (!batch.empty()) {
      auto callback = batch.begin();
      if (callback->ready && !callback->ready()) {
        std::lock_guard lock(mutex_);
        callbacks_.splice(callbacks_.end(), batch, callback);
        notify(); // Keep deferred work pollable for the owner's next pass.
      } else {
        callback->run(); // packaged_task returns exceptions to the worker.
        batch.erase(callback);
      }
    }
  }

  /// Run a blocking transport stop/join away from the owner while continuing
  /// to service device cleanup callbacks. The operation must join all of its
  /// producers before returning. Exceptions are rethrown only after joining.
  template <typename Function> void run_while_dispatching(Function &&function) {
    require_owner();
    std::atomic<bool> complete{false};
    std::packaged_task<void()> operation(std::forward<Function>(function));
    auto result = operation.get_future();
    std::jthread worker([&] {
      operation();
      complete.store(true, std::memory_order_release);
      notify();
    });
    while (!complete.load(std::memory_order_acquire)) {
      pollfd descriptor{fd(), POLLIN, 0};
      int status;
      do {
        status = ::poll(&descriptor, 1, -1);
      } while (status < 0 && errno == EINTR);
      if (status < 0 || (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)))
        std::terminate(); // Cannot safely join blocked producers without pumping.
      dispatch();
    }
    worker.join();
    dispatch(); // Consume the completion wakeup, including the fast-stop case.
    result.get();
  }

private:
  struct PendingCallback {
    std::function<bool()> ready;
    std::function<void()> run;
  };

  void require_owner() const {
    if (std::this_thread::get_id() != owner_)
      throw std::logic_error("embedded KMD callbacks require their owner thread");
  }

  void notify() noexcept {
    const uint64_t one = 1;
    while (::write(fd(), &one, sizeof(one)) < 0) {
      if (errno == EINTR)
        continue;
      if (errno != EAGAIN)
        std::terminate();
      break; // A saturated eventfd is already readable.
    }
  }

  const std::thread::id owner_ = std::this_thread::get_id();
  util::UniqueHandle notification_;
  std::mutex mutex_;
  std::list<PendingCallback> callbacks_;
};

} // namespace rocjitsu

#endif
