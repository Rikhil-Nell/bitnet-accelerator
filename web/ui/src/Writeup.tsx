import { For } from "solid-js";
import Arch from "./Arch";
import { board, bytesPerToken, milestones, model, profile } from "./data";

function TokenTrip() {
  const steps = [
    { x: 10, w: 110, t: "token id", s: "791 (\"The\")" },
    { x: 140, w: 130, t: "embedding", s: "2560 numbers" },
    { x: 290, w: 240, t: "30 × transformer layer", s: "attention + feed-forward" },
    { x: 550, w: 130, t: "LM head", s: "128,256 scores" },
    { x: 700, w: 110, t: "argmax", s: "next token" },
  ];
  return (
    <svg viewBox="0 0 820 150" role="img" aria-label="A token becomes a vector, passes through 30 layers, is scored against the vocabulary, and the best score is the next token">
      <For each={steps}>{(s, i) => (
        <g>
          <rect x={s.x} y="40" width={s.w} height="64" rx="10" fill="var(--surface-2)"
                stroke={i() === 2 ? "var(--accent)" : "var(--line)"} stroke-width={i() === 2 ? 2 : 1} />
          <text x={s.x + s.w / 2} y="68" text-anchor="middle" font-weight="600" font-size="14">{s.t}</text>
          <text x={s.x + s.w / 2} y="88" text-anchor="middle" class="muted" font-size="12.5">{s.s}</text>
          {i() < steps.length - 1 && (
            <path d={`M${s.x + s.w + 3} 72 l12 0 m-5 -5 l5 5 l-5 5`} fill="none" stroke="var(--muted)" stroke-width="1.6" />
          )}
        </g>
      )}</For>
      <path d="M760 106 C 760 140, 60 140, 60 106" fill="none" stroke="var(--muted)" stroke-dasharray="4 4" />
      <text x="410" y="146" text-anchor="middle" class="muted" font-size="12">the new token is appended and the loop runs again</text>
      <text x="410" y="26" text-anchor="middle" class="muted" font-size="12">each layer: 4 matrix-vector products (ternary weights) + small elementwise math</text>
    </svg>
  );
}

function BytesChart() {
  const max = 860;
  const w = (mb: number) => (mb / max) * 520;
  return (
    <svg viewBox="0 0 820 190" role="img" aria-label="Megabytes read from memory per generated token, by design version">
      <For each={bytesPerToken}>{(r, i) => {
        const y = 20 + i() * 52;
        const total = r.layers + r.head;
        return (
          <g>
            <text x="0" y={y + 14} font-size="12.5">{r.label}</text>
            <rect x="0" y={y + 22} width={w(r.layers)} height="18" rx="3" fill="var(--accent)" />
            <rect x={w(r.layers)} y={y + 22} width={w(r.head)} height="18" rx="3" fill="var(--accent-2)" />
            <text x={w(total) + 8} y={y + 36} font-size="12.5">{total} MB → {(total / 4.05).toFixed(0)} ms at 4.05 GB/s</text>
          </g>
        );
      }}</For>
      <rect x="560" y="2" width="12" height="12" fill="var(--accent)" /><text x="578" y="12" font-size="12">30 layers</text>
      <rect x="660" y="2" width="12" height="12" fill="var(--accent-2)" /><text x="678" y="12" font-size="12">LM head</text>
    </svg>
  );
}

function TernaryFig() {
  const xs = [37, -12, 85, 4, -66];
  const ws = [1, 0, -1, 1, 0];
  return (
    <svg viewBox="0 0 820 170" role="img" aria-label="With ternary weights a dot product is only additions and subtractions">
      <text x="0" y="20" font-weight="600" font-size="13">activations x (int8)</text>
      <text x="0" y="78" font-weight="600" font-size="13">weights w ∈ {"{-1, 0, +1}"}</text>
      <For each={xs}>{(x, i) => {
        const cx = 210 + i() * 92;
        const w = ws[i()];
        return (
          <g>
            <rect x={cx} y="4" width="70" height="26" rx="5" fill="var(--surface-2)" stroke="var(--line)" />
            <text x={cx + 35} y="22" text-anchor="middle" font-family="var(--mono)" font-size="13">{x}</text>
            <rect x={cx} y="62" width="70" height="26" rx="5" fill={w === 0 ? "var(--surface-2)" : "color-mix(in srgb, var(--accent) 20%, var(--surface-2))"} stroke="var(--line)" />
            <text x={cx + 35} y="80" text-anchor="middle" font-family="var(--mono)" font-size="13">{w > 0 ? "+1" : w}</text>
            <text x={cx + 35} y="116" text-anchor="middle" font-size="12.5" class="muted">{w === 1 ? `+${x}` : w === -1 ? `−(${x})` : "skip"}</text>
          </g>
        );
      }}</For>
      <text x="0" y="116" font-weight="600" font-size="13">what the hardware does</text>
      <text x="0" y="156" font-size="13">y = 37 − 85 + 4 = <tspan font-weight="700">−44</tspan>: no multipliers. One 64-bit word carries 40 weights (5 per byte, base 3), so a 40-input adder tree per unit consumes one word every clock.</text>
    </svg>
  );
}

function Pipeline() {
  const st = [
    ["AXI3 HP port", "16-beat bursts, 8 in flight"],
    ["512-deep FIFO", "credit-based, never stalls DDR"],
    ["T5 decode", "8 bytes → 40 weights"],
    ["40 lanes", "+x, −x or 0"],
    ["adder tree", "40 → 10 → 5 → 1"],
    ["accumulate", "one int32 per row"],
  ];
  return (
    <svg viewBox="0 0 820 110" role="img" aria-label="Engine unit pipeline">
      <For each={st}>{([t, s], i) => (
        <g>
          <rect x={4 + i() * 136} y="20" width="124" height="62" rx="8" fill="var(--surface-2)" stroke="var(--accent)" />
          <text x={66 + i() * 136} y="46" text-anchor="middle" font-size="13" font-weight="600">{t}</text>
          <text x={66 + i() * 136} y="64" text-anchor="middle" font-size="11" class="muted">{s}</text>
          {i() < st.length - 1 && <path d={`M${128 + i() * 136} 51 l10 0`} stroke="var(--muted)" stroke-width="1.6" />}
        </g>
      )}</For>
      <text x="410" y="104" text-anchor="middle" class="muted" font-size="12">one 64-bit word per clock per unit × 4 units × 142.9 MHz = 4.57 GB/s of capacity (DDR peak is 4.27 GB/s)</text>
    </svg>
  );
}

function Milestones() {
  const lo = Math.log10(100), hi = Math.log10(1e6);
  const x = (ms: number) => ((Math.log10(ms) - lo) / (hi - lo)) * 560;
  return (
    <svg viewBox="0 0 820 170" role="img" aria-label="Milliseconds per generated token over the project, log scale">
      <For each={milestones}>{(m, i) => (
        <g>
          <text x="0" y={22 + i() * 46} font-size="12.5">{m.step}</text>
          <rect x="0" y={30 + i() * 46} width={Math.max(4, x(m.ms))} height="14" rx="3" fill={i() === 0 ? "var(--muted)" : "var(--accent)"} />
          <text x={Math.max(4, x(m.ms)) + 8} y={42 + i() * 46} font-size="12.5" font-weight="600">
            {m.ms >= 1000 ? `${(m.ms / 1000).toFixed(0)} s` : `${m.ms} ms`} / token
          </text>
        </g>
      )}</For>
      <text x="0" y="165" class="muted" font-size="11.5">log scale: each bar step to the left is 10× faster</text>
    </svg>
  );
}

function SpecFig() {
  const box = (x: number, y: number, w: number, t: string, s: string, c = "var(--line)") => (
    <g>
      <rect x={x} y={y} width={w} height="46" rx="8" fill="var(--surface-2)" stroke={c} />
      <text x={x + w / 2} y={y + 20} text-anchor="middle" font-size="13" font-weight="600">{t}</text>
      <text x={x + w / 2} y={y + 37} text-anchor="middle" font-size="11.5" class="muted">{s}</text>
    </g>
  );
  return (
    <svg viewBox="0 0 820 200" role="img" aria-label="Speculative decoding with two positions per pass">
      {box(10, 20, 150, '"capital"', "current token")}
      {box(10, 100, 150, '"of"', "guess (n-gram table)", "var(--accent-2)")}
      <rect x="200" y="14" width="300" height="140" rx="12" fill="color-mix(in srgb, var(--accent) 10%, transparent)" stroke="var(--accent)" />
      <text x="350" y="40" text-anchor="middle" font-weight="700">one pass over 490 MB</text>
      <text x="350" y="62" text-anchor="middle" font-size="12" class="muted">every weight beat is multiplied</text>
      <text x="350" y="78" text-anchor="middle" font-size="12" class="muted">with both activation vectors</text>
      <text x="350" y="112" text-anchor="middle" font-size="12">position 2 attends to position 1</text>
      <path d="M160 43 L200 60 M160 123 L200 105" stroke="var(--muted)" fill="none" />
      {box(540, 20, 270, 'predicts "of"  ✓ = the guess', "so the 2nd output is valid too", "var(--accent-3)")}
      {box(540, 100, 270, 'predicts "France"', "2 tokens from 1 pass", "var(--accent-3)")}
      <path d="M500 50 L540 43 M500 110 L540 123" stroke="var(--muted)" fill="none" />
      <text x="410" y="190" text-anchor="middle" class="muted" font-size="12">if the first prediction differs from the guess, the second output is discarded: never a wrong answer, only wasted work</text>
    </svg>
  );
}

function ProfileBars() {
  const entries = Object.entries(profile).sort((a, b) => b[1] - a[1]);
  const total = entries.reduce((s, [, v]) => s + v, 0);
  return (
    <div class="bars">
      <For each={entries}>{([k, v]) => (
        <div class="bar">
          <span>{k.replace("_", " ")}</span>
          <div class="track"><div class="fill" style={{ width: `${(v / total) * 100}%`, background: k === "PL_wait" ? "var(--accent)" : "var(--accent-2)" }} /></div>
          <span class="v">{(v / 1000).toFixed(1)} ms</span>
        </div>
      )}</For>
    </div>
  );
}

export default function Writeup() {
  return (
    <article class="article">
      <div class="kicker">Capstone · FPGA inference</div>
      <h1>A 2-billion-parameter language model on a hobby FPGA board</h1>
      <p class="lede">
        Microsoft's {model.name} answers questions on a {board.name}: a $300 board with one gigabyte of
        memory. No PC in the loop for the math. This is how it works, written for electrical engineers
        who have never looked inside a language model.
      </p>

      <div class="stats">
        <div class="stat"><b>5.1</b><span>tokens / s (words-ish per second)</span></div>
        <div class="stat"><b>2.4 B</b><span>parameters, all on the board</span></div>
        <div class="stat"><b>95 %</b><span>of the DDR's peak bandwidth used</span></div>
        <div class="stat"><b>2,700×</b><span>faster than our first design</span></div>
      </div>

      <h2>1. What a language model actually computes</h2>
      <p>
        Forget the mystique: an LLM is a fixed pipeline of arithmetic. Text is cut into <em>tokens</em> (word pieces,
        here from a vocabulary of {model.vocab.toLocaleString()}). Each token id selects a row of a table, giving a
        vector of {model.hidden.toLocaleString()} numbers. That vector passes through {model.layers} identical
        layers. At the end, a final matrix (the <em>LM head</em>) gives one score per vocabulary entry, and the
        highest score is the next token. The new token is fed back in, and the loop repeats.
      </p>
      <figure class="figure wide"><TokenTrip /><figcaption>One generated token = one trip through this pipeline.</figcaption></figure>
      <p>Each layer does two things:</p>
      <ul>
        <li><b>Attention</b>: the vector is projected into a query, key and value (one matrix-vector product,
          3840 × 2560). The query is compared with the keys of every earlier token (kept in the <em>KV cache</em>),
          and the values are mixed according to those similarities. Another matrix (2560 × 2560) maps the result back.</li>
        <li><b>Feed-forward</b>: a 13824 × 2560 matrix expands the vector, a squared ReLU gates it, and a 2560 × 6912
          matrix shrinks it back.</li>
      </ul>
      <p>
        Between them sit small elementwise steps: RMS normalization (divide by the vector's RMS value),
        rotary position encoding, residual additions. In hardware terms: <b>almost all the work is
        matrix-vector multiplication, and every weight is used exactly once per token.</b>
      </p>

      <h2>2. Why speed is a memory problem</h2>
      <p>
        A matrix-vector product does one multiply-add per weight it reads. There is no reuse: to make one
        token, the chip must stream the entire model from DRAM. So the speed limit is not arithmetic, it is
        the memory bus. If you remember one equation from this page, make it this one:
      </p>
      <blockquote><b>tokens per second ≈ memory bandwidth ÷ bytes read per token</b></blockquote>
      <p>
        The Zybo's DDR3L is a 32-bit bus at 1066 MT/s: {board.ddrPeakGBs} GB/s at best. In roofline terms the
        workload's arithmetic intensity is about 5 operations per byte, far left of where a multiplier array
        would be the bottleneck. Every optimization below is about one of two things: <em>read fewer bytes</em>,
        or <em>use more of the bandwidth</em>.
      </p>
      <figure class="figure wide"><BytesChart /><figcaption>Bytes the engine reads per generated token, and the time that takes at the measured 4.05 GB/s.</figcaption></figure>

      <h2>3. BitNet's trick: weights that are −1, 0 or +1</h2>
      <p>
        Most models store weights as 16-bit floats. BitNet b1.58 was <em>trained</em> with every weight restricted
        to three values, times one scale per matrix. Activations are rounded to 8-bit integers. This does two
        things for us. The model shrinks to 1.58 bits of information per weight, so it fits in 1 GB. And a
        "multiply" becomes add, subtract or skip, which an FPGA does in a handful of lookup tables.
      </p>
      <figure class="figure wide"><TernaryFig /></figure>

      <h2>4. The hardware</h2>
      <p>
        The Zynq-7010 is two chips in one package: a dual-core ARM Cortex-A9 (the <em>processing system</em>, PS)
        and an FPGA (the <em>programmable logic</em>, PL), sharing the same DRAM. The PL reaches the DRAM through
        four 64-bit "high performance" AXI ports. We built one matrix-vector engine per port.
      </p>
      <figure class="figure wide"><Arch label="The PL streams weights; the ARM does everything that is not a big matrix." /></figure>
      <p>
        Each unit is a straight pipeline that consumes one 64-bit word per clock and never stalls on its own:
      </p>
      <figure class="figure wide"><Pipeline /></figure>
      <h3>Why AXI is not the bottleneck</h3>
      <p>
        "AXI" is a protocol, not a speed. Our first design fed the accelerator through the general-purpose
        AXI-Lite port: the CPU wrote values one 32-bit register at a time, while the PC drove it over a 115,200-baud
        UART. Now the PL is the bus <em>master</em>: it requests 16-beat bursts of 64-bit data, keeps 8 requests in
        flight per port, and uses all four ports at once. Four ports × 8 bytes × 142.9 MHz is 4.57 GB/s of capacity,
        above what the DRAM can deliver, so the DRAM is the limit. We measure 4.05 GB/s, 95 % of its peak. The
        missing 5 % is refresh and row switches inside the DRAM. AXI-Lite is still there, but only for 13 register
        writes per matrix.
      </p>

      <h2>5. The optimizations, in order</h2>
      <figure class="figure wide"><Milestones /></figure>
      <h3>Weights live in DDR; the PC only does text</h3>
      <p>The 782 MB weight image is loaded from the SD card at boot. The PC (or a phone, via this page) only turns text into token ids and back.</p>
      <h3>An engine sized to the memory, not bigger</h3>
      <p>
        4.27 GB/s ÷ 32 bytes per clock (4 units × 8 bytes) = 133 MHz: the minimum clock at which the engine
        keeps up with the DRAM. We closed timing at 142.9 MHz by pipelining the adder tree and registering the
        BRAM outputs. Using more logic would not help. That is why the FPGA is only 38 % full.
      </p>
      <h3>A DRAM bank collision</h3>
      <p>
        When the four units' slices started a multiple of 32 KB apart, all four hit the same DRAM bank at the
        same moment and bandwidth halved (2.25 vs 4.05 GB/s). The image now staggers the slices by 8 KB.
      </p>
      <h3>A cheaper LM head that gives the same answer</h3>
      <p>
        The final 128,256 × 2560 matrix was 39 % of all bytes. A ternary copy of it (82 MB instead of 328 MB) is
        good enough to <em>rank</em> the vocabulary: over 250 test steps, the correct next token was always in its
        top 16. So the engine ranks with the cheap copy, and the ARM rescores the top 64 exactly in int8.
        The answer is unchanged and the head is 4× cheaper.
      </p>
      <h3>1.6-bit packing</h3>
      <p>
        Three values need log₂3 = 1.58 bits, but 2-bit storage wastes 21 %. Since 3⁵ = 243 ≤ 256, five
        weights fit in one byte as a base-3 number. Decoding is a 256-entry lookup per byte in the fabric.
        20 % fewer bytes means 20 % less time.
      </p>
      <h3>Keeping the ARM out of the way</h3>
      <p>
        Everything that is not a big matrix runs on the ARM: normalization, quantization, attention, picking the
        next token. Rewritten with NEON SIMD it dropped from 80 ms to about 40 ms per token. The engine also
        exposes per-unit progress counters, so the ARM processes finished rows <em>while</em> the FPGA streams
        the rest of the matrix.
      </p>

      <h2>6. Where the time goes now</h2>
      <p>Per generated token (short answer), as measured by the firmware's profiler:</p>
      <div class="figure"><ProfileBars /></div>
      <p>
        "PL wait" is the FPGA streaming weights at the memory limit. Attention grows with the length of the
        conversation, since every new token looks back at all previous ones.
      </p>

      <h2>7. Two tokens per pass: speculative decoding</h2>
      <p>
        Once the engine runs at the memory limit, a single stream of tokens cannot go faster than one pass over
        490 MB of weights per token: about 8.3 passes per second. The only lever left is to get <em>more than one
        token out of each pass</em>.
      </p>
      <p>
        The engine's third version multiplies every weight it reads with <b>two</b> activation vectors instead
        of one: a second 40-lane adder tree per unit, fed by the same decoded weights, costing extra logic but no
        extra memory bandwidth. We feed it the current token and a <em>guess</em> for the token after it. The model
        computes its real prediction for both positions in the same pass. If the real prediction equals the
        guess, both outputs are valid and the pass produced two tokens. If not, we keep the first and throw the
        second away. The output is exactly what the model would have produced one token at a time.
      </p>
      <figure class="figure wide"><SpecFig /><figcaption>One pass, two positions. The guess only has to be right often enough to pay for the extra ARM work of the second position.</figcaption></figure>
      <p>
        The guesses come from two cheap sources: the conversation itself ("the token that followed these two
        tokens earlier"), and a table of the model's own habits, built by letting the model answer a few thousand
        questions on a GPU and recording which token most often follows each short sequence. On questions the
        table never saw, about one pass in five produces two tokens.
      </p>

      <h2>8. How far can it go?</h2>
      <p>
        The DDR sets a hard floor: about 121 ms to stream the weights once. Everything else is either hidden
        behind that stream (epilogues, the vocabulary ranking) or split across both ARM cores (attention).
        What is left is the speculation hit rate, which grows slowly with more drafting data.
      </p>
      <p><a href="#/demo">Try it live →</a></p>
    </article>
  );
}
