// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "random_seed.h"
#include "random_generator.h"
#include <atomic>
#include <chrono>

#include <shared_mutex>
#include <unordered_map>
#include <array>
#include <algorithm>

namespace onnxruntime {
namespace utils {

// "Global initializer calls a non-constexpr function."
//TODO: Fix the warning. The variable should be put in the environment class.
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning(disable : 26426)
#endif
static std::atomic<int64_t> g_random_seed(std::chrono::system_clock::now().time_since_epoch().count());


#ifdef OPENVPI_ORTDIST_PATCH
static std::atomic<int64_t> g_openvpi_current_session_id;
static std::shared_mutex g_openvpi_session_seed_lock;
static std::unordered_map<int64_t, SessionSpec> g_openvpi_session_seed_map;

constexpr std::array g_nonDeterministicOps{"RandomUniform", "RandomNormal", "RandomUniformLike", "RandomNormalLike", "Multinomial"};
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif

int64_t GetRandomSeed() {
  return g_random_seed.load();
}

void SetRandomSeed(int64_t seed) {
  g_random_seed.store(seed);

  // Reset default generators.
  RandomGenerator::Default().SetSeed(seed);
  PhiloxGenerator::Default().SetSeed(static_cast<uint64_t>(seed));
}

#ifdef OPENVPI_ORTDIST_PATCH
enum class OpenVPIRequestType {
    SetCurrentSessionId = 0,
    GetCurrentSessionId,
    SetSessionSeed,
    GetSessionSeed,
    SetSessionTaskId,
    GetSessionTaskId,
    RemoveSession,
};

bool isOpNondeterministic(const std::string &op) {
  return std::find(g_nonDeterministicOps.begin(), g_nonDeterministicOps.end(), op) != g_nonDeterministicOps.end();
}

const char* SessionIdAttributeName = "__openvpi_session_id__";

int64_t GetCurrentSessionId() {
  return g_openvpi_current_session_id.load();
}

SessionSpec GetSessionSpec(int64_t key) {
  std::shared_lock<std::shared_mutex> lock(g_openvpi_session_seed_lock);
  auto it = g_openvpi_session_seed_map.find(key);
  if (it == g_openvpi_session_seed_map.end()) {
    return {};
  } else {
    return it->second;
  }
}

void AccessOpenVPIRandomSeed(int type, int64_t key, int64_t value, int64_t* out) {
  switch ((OpenVPIRequestType) type) {
    case OpenVPIRequestType::SetCurrentSessionId:
      g_openvpi_current_session_id.store(value);
      break;
    case OpenVPIRequestType::GetCurrentSessionId:
      *out = g_openvpi_current_session_id.load();
      break;
    case OpenVPIRequestType::SetSessionSeed: {
      std::unique_lock<std::shared_mutex> lock(g_openvpi_session_seed_lock);
      auto it = g_openvpi_session_seed_map.find(key);
      if (it == g_openvpi_session_seed_map.end()) {
        g_openvpi_session_seed_map.insert(std::make_pair(key, SessionSpec(value)));
      } else {
        it->second.seed = value;
      }
      break;
    }
    case OpenVPIRequestType::GetSessionSeed: {
      std::shared_lock<std::shared_mutex> lock(g_openvpi_session_seed_lock);
      auto it = g_openvpi_session_seed_map.find(key);
      if (it == g_openvpi_session_seed_map.end()) {
        *out = value;
      } else {
        *out = it->second.seed;
      }
      break;
    }
    case OpenVPIRequestType::SetSessionTaskId: {
      std::unique_lock<std::shared_mutex> lock(g_openvpi_session_seed_lock);
      auto it = g_openvpi_session_seed_map.find(key);
      if (it == g_openvpi_session_seed_map.end()) {
        g_openvpi_session_seed_map.insert(std::make_pair(key, SessionSpec(0, value)));
      } else {
        it->second.taskId = value;
      }
      break;
    }
    case OpenVPIRequestType::GetSessionTaskId: {
      std::shared_lock<std::shared_mutex> lock(g_openvpi_session_seed_lock);
      auto it = g_openvpi_session_seed_map.find(key);
      if (it == g_openvpi_session_seed_map.end()) {
        *out = value;
      } else {
        *out = it->second.taskId;
      }
      break;
    }
    case OpenVPIRequestType::RemoveSession: {
      std::unique_lock<std::shared_mutex> lock(g_openvpi_session_seed_lock);
      auto it = g_openvpi_session_seed_map.find(key);
      if (it != g_openvpi_session_seed_map.end()) {
        g_openvpi_session_seed_map.erase(it);
      }
      break;
    }
    default:
      break;
  }
}
#endif

}  // namespace utils
}  // namespace onnxruntime
