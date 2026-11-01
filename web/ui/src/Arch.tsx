import { For } from "solid-js";

// System diagram: DDR -> 4 HP ports -> 4 engine units (PL), Cortex-A9 (PS) on the side.
// `pl` / `arm` light up the parts that are working (the live demo drives them).
export default function Arch(props: { pl?: boolean; arm?: boolean; label?: string }) {
  const units = [0, 1, 2, 3];
  const uy = (u: number) => 52 + u * 62;
  return (
    <svg viewBox="0 0 820 330" role="img" aria-label="System architecture: DDR memory feeds four engine units in the FPGA over four HP ports; the ARM cores run attention and control">
      <style>{`
        .flow { stroke-dasharray: 6 8; }
        .flow.on { animation: dash 0.6s linear infinite; }
        @keyframes dash { to { stroke-dashoffset: -14; } }
        .box { stroke: var(--line); fill: var(--surface-2); }
        .pl.on { fill: color-mix(in srgb, var(--accent) 22%, var(--surface-2)); stroke: var(--accent); }
        .ps.on { fill: color-mix(in srgb, var(--accent-2) 22%, var(--surface-2)); stroke: var(--accent-2); }
      `}</style>
      {/* DDR */}
      <rect x="16" y="40" width="150" height="250" rx="10" class="box" style={{ stroke: "var(--accent-3)" }} />
      <text x="91" y="70" text-anchor="middle" font-weight="700">DDR3L</text>
      <text x="91" y="90" text-anchor="middle" class="muted" font-size="13">1 GB, 4.27 GB/s peak</text>
      <rect x="32" y="108" width="118" height="92" rx="6" fill="color-mix(in srgb, var(--accent-3) 18%, transparent)" />
      <text x="91" y="140" text-anchor="middle" font-size="13">weights</text>
      <text x="91" y="158" text-anchor="middle" font-size="13">782 MB</text>
      <text x="91" y="176" text-anchor="middle" class="muted" font-size="12">1.6-bit ternary</text>
      <rect x="32" y="212" width="118" height="60" rx="6" fill="color-mix(in srgb, var(--accent-2) 14%, transparent)" />
      <text x="91" y="238" text-anchor="middle" font-size="13">KV cache</text>
      <text x="91" y="256" text-anchor="middle" class="muted" font-size="12">fp16, per token</text>

      {/* HP ports + units */}
      <rect x="300" y="24" width="270" height="282" rx="12" fill="none" stroke="var(--accent)" stroke-dasharray="4 4" />
      <text x="435" y="18" text-anchor="middle" font-size="12" fill="var(--accent)" font-weight="600">FPGA fabric (PL), 143 MHz</text>
      <For each={units}>{(u) => (
        <g>
          <line x1="166" y1={uy(u) + 22} x2="318" y2={uy(u) + 22} stroke="var(--accent)" stroke-width="3"
                class={`flow ${props.pl ? "on" : ""}`} />
          <text x="242" y={uy(u) + 14} text-anchor="middle" class="muted" font-size="11">HP{u} 64-bit AXI</text>
          <rect x="318" y={uy(u)} width="234" height="44" rx="8" class={`box pl ${props.pl ? "on" : ""}`} />
          <text x="330" y={uy(u) + 19} font-size="13" font-weight="600">GEMV unit {u}</text>
          <text x="330" y={uy(u) + 35} font-size="11.5" class="muted">FIFO → T5 decode → 40 lanes → adder tree</text>
        </g>
      )}</For>

      {/* PS */}
      <rect x="634" y="40" width="170" height="250" rx="10" class={`box ps ${props.arm ? "on" : ""}`} />
      <text x="719" y="70" text-anchor="middle" font-weight="700">Cortex-A9 ×2</text>
      <text x="719" y="90" text-anchor="middle" class="muted" font-size="13">ARM (PS), 667 MHz</text>
      <text x="719" y="130" text-anchor="middle" font-size="12.5">attention</text>
      <text x="719" y="150" text-anchor="middle" font-size="12.5">RMSNorm, quantize</text>
      <text x="719" y="170" text-anchor="middle" font-size="12.5">ReLU², residual</text>
      <text x="719" y="190" text-anchor="middle" font-size="12.5">top-64 rescoring</text>
      <text x="719" y="228" text-anchor="middle" class="muted" font-size="11.5">AXI-Lite: 13 register</text>
      <text x="719" y="243" text-anchor="middle" class="muted" font-size="11.5">writes per matrix</text>
      <line x1="634" y1="250" x2="570" y2="250" stroke="var(--accent-2)" stroke-dasharray="3 3" />
      <line x1="552" y1="160" x2="634" y2="160" stroke="var(--accent-2)" stroke-width="2" />
      <text x="593" y="152" text-anchor="middle" class="muted" font-size="11">results</text>
      {props.label && <text x="410" y="325" text-anchor="middle" class="muted" font-size="12">{props.label}</text>}
    </svg>
  );
}
