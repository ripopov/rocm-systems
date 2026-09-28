// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_KMD_LINUX_DOORBELL_MONITOR_H_
#define ROCJITSU_KMD_LINUX_DOORBELL_MONITOR_H_

#include "rocjitsu/kmd/server.h"
#include "util/unique_handle.h"

#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <map>
#include <mutex>
#include <sys/eventfd.h>
#include <system_error>
#include <thread>

namespace rocjitsu {

/// Watches common KFD's private doorbell mappings, never client aliases. The
/// mapping is borrowed until remove()/remove_process() returns. Queue storage
/// is reserved before host queue creation, then activated without allocation.
/// Notifications carry the latest producer pointer per queue (not a count of
/// writes); storage remains bounded by the number of live/reserved queues.
class DoorbellMonitor {
public:
  DoorbellMonitor() : notification_(::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK)) {
    if (!notification_)
      throw std::system_error(errno, std::generic_category(), "doorbell eventfd");
    worker_ = std::jthread([this](std::stop_token stop) {
      std::unique_lock lock(mutex_);
      while (!stop.stop_requested()) {
        sample_locked();
        changed_.wait_for(lock, std::chrono::microseconds(250),
                          [&] { return stop.stop_requested(); });
      }
    });
  }

  ~DoorbellMonitor() {
    worker_.request_stop();
    changed_.notify_all();
    worker_.join();
  }

  /// Borrowed poll-only descriptor; drain() consumes its readiness.
  int fd() const { return notification_.get(); }

  int reserve(uint32_t process, uint32_t queue, uint32_t offset) {
    if (offset % alignof(uint64_t))
      return -EINVAL;
    std::lock_guard lock(mutex_);
    auto [it, inserted] = queues_.try_emplace(key(process, queue));
    if (!inserted)
      return -EEXIST;
    it->second.record = {RJ_KMD_SERVER_ABI_VERSION,
                         sizeof(rj_kmd_doorbell_record_t),
                         process,
                         queue,
                         offset,
                         0,
                         ~uint64_t(0)};
    return 0;
  }

  /// The caller guarantees that base+offset addresses a readable uint64_t in
  /// the private mapping. A null base defers sampling until set_base().
  void activate(uint32_t process, uint32_t queue, const void *base) {
    std::lock_guard lock(mutex_);
    auto &watch = queues_.at(key(process, queue));
    watch.active = true;
    watch.base = static_cast<const uint8_t *>(base);
  }

  void set_base(uint32_t process, const void *base) {
    std::lock_guard lock(mutex_);
    for (auto &[id, watch] : queues_)
      if (watch.record.process_id == process)
        watch.base = static_cast<const uint8_t *>(base);
  }

  void remove(uint32_t process, uint32_t queue) {
    std::lock_guard lock(mutex_);
    queues_.erase(key(process, queue));
    refresh_notification_locked();
  }

  void remove_process(uint32_t process) {
    std::lock_guard lock(mutex_);
    std::erase_if(queues_,
                  [&](const auto &entry) { return entry.second.record.process_id == process; });
    refresh_notification_locked();
  }

  size_t pending() const {
    std::lock_guard lock(mutex_);
    return pending_locked();
  }

  bool contains(uint32_t process, uint32_t queue) const {
    std::lock_guard lock(mutex_);
    return queues_.contains(key(process, queue));
  }

  size_t active() const {
    std::lock_guard lock(mutex_);
    size_t count = 0;
    for (const auto &[id, watch] : queues_)
      count += watch.active;
    return count;
  }

  size_t drain(rj_kmd_doorbell_record_t *records, size_t capacity) {
    std::lock_guard lock(mutex_);
    size_t count = 0;
    for (auto &[id, watch] : queues_) {
      if (count == capacity)
        break;
      if (watch.pending) {
        records[count++] = watch.record;
        watch.pending = false;
      }
    }
    refresh_notification_locked();
    return count;
  }

private:
  struct Watch {
    rj_kmd_doorbell_record_t record{};
    const uint8_t *base = nullptr;
    bool active = false;
    bool pending = false;
  };

  static uint64_t key(uint32_t process, uint32_t queue) {
    return (uint64_t(process) << 32) | queue;
  }

  size_t pending_locked() const {
    size_t count = 0;
    for (const auto &[id, watch] : queues_)
      count += watch.pending;
    return count;
  }

  void notify_locked() const noexcept {
    const uint64_t one = 1;
    while (::write(fd(), &one, sizeof(one)) < 0) {
      if (errno == EINTR)
        continue;
      if (errno != EAGAIN)
        std::terminate(); // The owner must not close the borrowed descriptor.
      break;
    }
  }

  void refresh_notification_locked() const noexcept {
    uint64_t value;
    while (::read(fd(), &value, sizeof(value)) < 0) {
      if (errno == EINTR)
        continue;
      if (errno != EAGAIN)
        std::terminate();
      break;
    }
    if (pending_locked())
      notify_locked();
  }

  void sample_locked() {
    bool changed = false;
    for (auto &[id, watch] : queues_) {
      if (!watch.active || !watch.base)
        continue;
      const auto *address = reinterpret_cast<const uint64_t *>(watch.base + watch.record.offset);
      const auto value = __atomic_load_n(address, __ATOMIC_ACQUIRE);
      if (value != watch.record.value) {
        watch.record.value = value;
        watch.pending = true;
        changed = true;
      }
    }
    if (changed)
      notify_locked();
  }

  util::UniqueHandle notification_;
  mutable std::mutex mutex_;
  std::condition_variable changed_;
  std::map<uint64_t, Watch> queues_;
  std::jthread worker_;
};

} // namespace rocjitsu

#endif
