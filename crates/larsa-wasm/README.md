# larsa-wasm

WebAssembly build of SRFM physics for browser-based interactive demos.

## Build
```bash
# Install wasm-pack
cargo install wasm-pack

# Build
CARGO_HOME=C:/Users/Matthew/.cargo wasm-pack build --target web
```

## Output
`pkg/larsa_wasm.js` + `pkg/larsa_wasm_bg.wasm`

## Usage in HTML
```html
<script type="module">
import init, { SRFMState, simulate_series } from './pkg/larsa_wasm.js';
await init();
const state = new SRFMState(0.005, 1.5, 0.95, 5000.0);
const signal = state.step(5010.0);  // returns position fraction
</script>
```
