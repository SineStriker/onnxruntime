// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <string>

namespace onnxruntime {
namespace utils {

/**
 * Gets the random seed value used by onnxruntime.
 *
 * The random seed value can be override with SetRandomSeed().
 *
 * @return The random seed value.
 */
int64_t GetRandomSeed();

/**
 * Sets the random seed value to be used by onnxruntime.
 *
 * If not called manually, the current clock will be used.
 *
 * @param seed The random seed value to use.
 */
void SetRandomSeed(int64_t seed);

// OpenVPI Addition
bool isOpNondeterministic(const std::string& op);

extern const char* SessionIdAttributeName;

int64_t GetCurrentSessionId();

struct SessionSpec {
  int64_t seed;
  int64_t taskId;

  SessionSpec() : SessionSpec(0){};
  SessionSpec(int64_t seed) : SessionSpec(seed, 0){};
  SessionSpec(int64_t seed, int64_t taskId) : seed(seed), taskId(taskId){};
};

SessionSpec GetSessionSpec(int64_t key);

void AccessOpenVPIRandomSeed(int type, int64_t key, int64_t value, int64_t *out);

}  // namespace utils
}  // namespace onnxruntime