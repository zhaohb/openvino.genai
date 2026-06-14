// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the OpenVINO GenAI C API, which is a C wrapper for the
 * ov::genai::Adapter and ov::genai::AdapterConfig classes (LoRA support).
 *
 * @file lora_adapter.h
 */

#pragma once

#include <stddef.h>

#include "openvino/c/ov_common.h"
#include "openvino/genai/c/visibility.h"

/**
 * @struct ov_genai_adapter
 * @brief type define ov_genai_adapter from ov_genai_adapter_opaque
 *
 * An immutable LoRA Adapter that carries the adaptation matrices and serves as a
 * unique adapter identifier.
 */
typedef struct ov_genai_adapter_opaque ov_genai_adapter;

/**
 * @struct ov_genai_adapter_config
 * @brief type define ov_genai_adapter_config from ov_genai_adapter_config_opaque
 *
 * A config that defines a combination of LoRA adapters with blending parameters
 * (alpha) and an application mode.
 */
typedef struct ov_genai_adapter_config_opaque ov_genai_adapter_config;

/**
 * @brief Application mode for an adapter config. Mirrors ov::genai::AdapterConfig::Mode.
 *        MODE_AUTO        - selected automatically depending on device/usage.
 *        MODE_DYNAMIC     - A, B, alpha fully variable (allows per-generate switching).
 *        MODE_STATIC_RANK - A, B static shape, alpha variable.
 *        MODE_STATIC      - A, B, alpha constants (keeps weights precision).
 *        MODE_FUSE        - A, B, alpha constants fused into the main matrix W (fastest, no switching).
 */
typedef enum {
    OV_GENAI_ADAPTER_MODE_AUTO = 0,
    OV_GENAI_ADAPTER_MODE_DYNAMIC = 1,
    OV_GENAI_ADAPTER_MODE_STATIC_RANK = 2,
    OV_GENAI_ADAPTER_MODE_STATIC = 3,
    OV_GENAI_ADAPTER_MODE_FUSE = 4
} ov_genai_adapter_mode_e;

// ─── Adapter ────────────────────────────────────────────────────────────────

/**
 * @brief Create an Adapter from a LoRA adapter file in safetensors format.
 * @param file_path Path to the adapter file (.safetensors).
 * @param adapter A pointer to the newly created ov_genai_adapter.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_adapter_create(const char* file_path, ov_genai_adapter** adapter);

/**
 * @brief Release the memory allocated by ov_genai_adapter.
 * @param adapter A pointer to the ov_genai_adapter to free.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_adapter_free(ov_genai_adapter* adapter);

// ─── AdapterConfig ──────────────────────────────────────────────────────────

/**
 * @brief Create an empty AdapterConfig with the given mode.
 * @param mode The application mode (see ov_genai_adapter_mode_e).
 * @param config A pointer to the newly created ov_genai_adapter_config.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_adapter_config_create(ov_genai_adapter_mode_e mode,
                                                                    ov_genai_adapter_config** config);

/**
 * @brief Release the memory allocated by ov_genai_adapter_config.
 * @param config A pointer to the ov_genai_adapter_config to free.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_adapter_config_free(ov_genai_adapter_config* config);

/**
 * @brief Add an adapter with a given alpha blending coefficient to the config.
 * @param config A pointer to the ov_genai_adapter_config.
 * @param adapter A pointer to the ov_genai_adapter to add.
 * @param alpha The blending coefficient for this adapter.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_adapter_config_add(ov_genai_adapter_config* config,
                                                                 const ov_genai_adapter* adapter,
                                                                 float alpha);

/**
 * @brief Set the alpha blending coefficient for an adapter already in the config.
 * @param config A pointer to the ov_genai_adapter_config.
 * @param adapter A pointer to the ov_genai_adapter.
 * @param alpha The new blending coefficient.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_adapter_config_set_alpha(ov_genai_adapter_config* config,
                                                                       const ov_genai_adapter* adapter,
                                                                       float alpha);

/**
 * @brief Remove an adapter from the config.
 * @param config A pointer to the ov_genai_adapter_config.
 * @param adapter A pointer to the ov_genai_adapter to remove.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_adapter_config_remove(ov_genai_adapter_config* config,
                                                                    const ov_genai_adapter* adapter);
