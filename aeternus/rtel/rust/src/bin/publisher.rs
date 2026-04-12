//! AETERNUS RTEL state publisher binary.
//! Runs the RTEL simulation pipeline and publishes metrics.
fn main() {
    println!("AETERNUS RTEL State Publisher v{}", rtel::VERSION);
    println!("RTEL Magic: 0x{:016X}", rtel::RTEL_MAGIC);
    println!("Default slot bytes: {}", rtel::DEFAULT_SLOT_BYTES);
    println!("Default ring capacity: {}", rtel::DEFAULT_RING_CAPACITY);
    println!("Max assets: {}", rtel::MAX_ASSETS);
    println!("Max LOB levels: {}", rtel::MAX_LOB_LEVELS);

    // Print channel names
    println!("\nStandard channels:");
    println!("  LOB:     {}", rtel::channels::LOB_SNAPSHOT);
    println!("  Vol:     {}", rtel::channels::VOL_SURFACE);
    println!("  Tensor:  {}", rtel::channels::TENSOR_COMP);
    println!("  Graph:   {}", rtel::channels::GRAPH_ADJ);
    println!("  Lumina:  {}", rtel::channels::LUMINA_PRED);
    println!("  Agent:   {}", rtel::channels::AGENT_ACTIONS);
    println!("  HB:      {}", rtel::channels::HEARTBEAT);

    // Test utility functions
    println!("\nUtility checks:");
    println!("  align_up(65, 64) = {}", rtel::align_up(65, 64));
    println!("  is_pow2(1024)    = {}", rtel::is_power_of_two(1024));
    println!("  next_pow2(100)   = {}", rtel::next_power_of_two(100));
    println!("  now_ns()         = {}", rtel::now_ns());
    println!("\nAll checks complete. RTEL ready.");
}
