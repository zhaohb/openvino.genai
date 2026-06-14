// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/c/lora_adapter.h"

#include <filesystem>

#include "openvino/genai/lora_adapter.hpp"
#include "types_c.h"

namespace {
// Translate the C enum to the C++ AdapterConfig::Mode.
ov::genai::AdapterConfig::Mode to_cpp_mode(ov_genai_adapter_mode_e mode) {
    switch (mode) {
    case OV_GENAI_ADAPTER_MODE_DYNAMIC:
        return ov::genai::AdapterConfig::Mode::MODE_DYNAMIC;
    case OV_GENAI_ADAPTER_MODE_STATIC_RANK:
        return ov::genai::AdapterConfig::Mode::MODE_STATIC_RANK;
    case OV_GENAI_ADAPTER_MODE_STATIC:
        return ov::genai::AdapterConfig::Mode::MODE_STATIC;
    case OV_GENAI_ADAPTER_MODE_FUSE:
        return ov::genai::AdapterConfig::Mode::MODE_FUSE;
    case OV_GENAI_ADAPTER_MODE_AUTO:
    default:
        return ov::genai::AdapterConfig::Mode::MODE_AUTO;
    }
}
}  // namespace

// ─── Adapter ────────────────────────────────────────────────────────────────

ov_status_e ov_genai_adapter_create(const char* file_path, ov_genai_adapter** adapter) {
    if (!file_path || !adapter) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_adapter> _adapter = std::make_unique<ov_genai_adapter>();
        _adapter->object = std::make_shared<ov::genai::Adapter>(std::filesystem::path(file_path));
        *adapter = _adapter.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

void ov_genai_adapter_free(ov_genai_adapter* adapter) {
    if (adapter) {
        delete adapter;
    }
}

// ─── AdapterConfig ──────────────────────────────────────────────────────────

ov_status_e ov_genai_adapter_config_create(ov_genai_adapter_mode_e mode, ov_genai_adapter_config** config) {
    if (!config) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_adapter_config> _config = std::make_unique<ov_genai_adapter_config>();
        _config->object = std::make_shared<ov::genai::AdapterConfig>(to_cpp_mode(mode));
        *config = _config.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

void ov_genai_adapter_config_free(ov_genai_adapter_config* config) {
    if (config) {
        delete config;
    }
}

ov_status_e ov_genai_adapter_config_add(ov_genai_adapter_config* config,
                                        const ov_genai_adapter* adapter,
                                        float alpha) {
    if (!config || !(config->object) || !adapter || !(adapter->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->add(*(adapter->object), alpha);
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_adapter_config_set_alpha(ov_genai_adapter_config* config,
                                              const ov_genai_adapter* adapter,
                                              float alpha) {
    if (!config || !(config->object) || !adapter || !(adapter->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->set_alpha(*(adapter->object), alpha);
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_adapter_config_remove(ov_genai_adapter_config* config, const ov_genai_adapter* adapter) {
    if (!config || !(config->object) || !adapter || !(adapter->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        config->object->remove(*(adapter->object));
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
