// =============================================================================
// AETERNUS Real-Time Execution Layer (RTEL)
// rtel.hpp — Master include / convenience header
// =============================================================================
// Include this header to get access to the full RTEL API.
// =============================================================================

#pragma once

#include "shm_bus.hpp"
#include "ring_buffer.hpp"
#include "global_state_registry.hpp"
#include "module_wrapper.hpp"
#include "scheduler.hpp"
#include "latency_monitor.hpp"
#include "serialization.hpp"

namespace aeternus::rtel {

// ---------------------------------------------------------------------------
// RTEL version info
// ---------------------------------------------------------------------------
static constexpr int kVersionMajor = 0;
static constexpr int kVersionMinor = 1;
static constexpr int kVersionPatch = 0;
static constexpr const char* kVersionString = "0.1.0";

// ---------------------------------------------------------------------------
// Quick-start: Initialize everything with one call
// ---------------------------------------------------------------------------
struct RTELInitOptions {
    bool create_channels    = true;
    bool register_modules   = true;
    bool start_watchdog     = true;
    bool verbose            = false;
    int  n_scheduler_workers= 4;
    bool pin_cpus           = false;
};

// Initialize RTEL with default configuration
inline bool initialize(const RTELInitOptions& opts = {}) {
    if (opts.verbose) {
        std::printf("[RTEL] Initializing AETERNUS RTEL v%s\n", kVersionString);
    }
    // Create ShmBus channels
    if (opts.create_channels) {
        ShmBus::instance().create_aeternus_channels();
        if (opts.verbose) std::printf("[RTEL] ShmBus channels created\n");
    }
    // Register all modules
    if (opts.register_modules) {
        auto& reg = ModuleRegistry::instance();
        reg.register_module(std::make_unique<ChronosWrapper>());
        reg.register_module(std::make_unique<NeuroSDEWrapper>());
        reg.register_module(std::make_unique<TensorNetWrapper>());
        reg.register_module(std::make_unique<OmniGraphWrapper>());
        reg.register_module(std::make_unique<LuminaWrapper>());
        reg.register_module(std::make_unique<HyperAgentWrapper>());
        bool ok = reg.initialize_all();
        if (opts.verbose) std::printf("[RTEL] Modules initialized: %s\n", ok?"OK":"FAIL");
        if (!ok) return false;
    }
    // Set up latency monitor
    LatencyMonitor::instance().set_violation_callback([](const SLAViolation& v) {
        std::fprintf(stderr, "[RTEL SLA] Stage %s exceeded by %luµs\n",
                     stage_name(v.stage), v.overage_ns / 1000);
    });
    // Register standard schemas
    SchemaRegistry::instance().register_aeternus_schemas();

    if (opts.verbose) std::printf("[RTEL] Initialization complete.\n");
    return true;
}

// Shutdown everything
inline void shutdown() {
    ModuleRegistry::instance().shutdown_all();
    ShmBus::instance().shutdown();
}

} // namespace aeternus::rtel
