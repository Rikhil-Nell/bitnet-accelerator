import { createSignal, For, onCleanup, onMount, Show } from "solid-js";
import Arch from "./Arch";

type Msg = { role: "user" | "bot"; text: string; meta?: string };
type Cand = { texts: string[]; logits: number[] };

const SUGGEST = [
  "What is the capital of France?",
  "How many legs does a spider have?",
  "Who wrote Romeo and Juliet?",
  "Explain in two sentences why the sky is blue.",
];

const COLORS: Record<string, string> = { PL_wait: "var(--accent)", attention: "var(--accent-2)" };

export default function Demo() {
  const [msgs, setMsgs] = createSignal<Msg[]>([]);
  const [input, setInput] = createSignal("");
  const [busy, setBusy] = createSignal(false);
  const [online, setOnline] = createSignal<boolean | null>(null);
  const [multi, setMulti] = createSignal(false);
  const [phase, setPhase] = createSignal<"idle" | "prefill" | "decode">("idle");
  const [prof, setProf] = createSignal<Record<string, number>>({});
  const [cand, setCand] = createSignal<Cand | null>(null);
  const [tokMs, setTokMs] = createSignal<number[]>([]);
  let ws: WebSocket | undefined;
  let box: HTMLDivElement | undefined;

  const connect = () => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.onopen = () => setOnline(true);
    ws.onclose = () => { setOnline(false); setBusy(false); setTimeout(connect, 2000); };
    ws.onmessage = (e) => onEvent(JSON.parse(e.data));
  };

  const appendBot = (text: string) => {
    setMsgs((m) => {
      const last = m[m.length - 1];
      if (last && last.role === "bot" && !last.meta) return [...m.slice(0, -1), { ...last, text: last.text + text }];
      return [...m, { role: "bot", text }];
    });
    queueMicrotask(() => box && (box.scrollTop = box.scrollHeight));
  };

  const onEvent = (ev: any) => {
    if (ev.type === "token") {
      setPhase("decode");
      appendBot(ev.text);
      if (ev.ms) setTokMs((t) => [...t.slice(-59), ev.ms]);
    } else if (ev.type === "profile") {
      if (ev.phase === "decode") setProf(ev.us);
    } else if (ev.type === "cand") {
      setCand({ texts: ev.texts, logits: ev.logits });
    } else if (ev.type === "done") {
      const steps = Math.max(ev.n_gen - 1, 1);
      const tps = ev.decode_ms ? (steps * 1000) / ev.decode_ms : 0;
      setMsgs((m) => {
        const last = m[m.length - 1];
        const meta = `${ev.n_prompt} prompt tokens in ${(ev.prefill_ms / 1000).toFixed(1)} s · ${ev.n_gen} tokens · ${tps.toFixed(2)} tok/s`;
        return last && last.role === "bot" ? [...m.slice(0, -1), { ...last, meta }] : m;
      });
      setBusy(false);
      setPhase("idle");
    } else if (ev.type === "error") {
      appendBot(`[board error] ${ev.message}`);
      setBusy(false);
      setPhase("idle");
    }
  };

  const ask = (q?: string) => {
    const text = (q ?? input()).trim();
    if (!text || busy() || !ws || ws.readyState !== 1) return;
    setMsgs((m) => [...m, { role: "user", text }, { role: "bot", text: "" }]);
    setInput("");
    setBusy(true);
    setPhase("prefill");
    setCand(null);
    ws.send(JSON.stringify({ type: "ask", prompt: text, multi: multi(), max: 64 }));
  };

  onMount(connect);
  onCleanup(() => ws?.close());

  const profEntries = () => Object.entries(prof()).filter(([k]) => k !== "total").sort((a, b) => b[1] - a[1]);
  const profTotal = () => prof().total || profEntries().reduce((s, [, v]) => s + v, 0);

  return (
    <div class="demo">
      <section class="panel chat">
        <h3>Ask the board</h3>
        <div class="row">
          <span class="pill"><span class={`dot ${online() ? "on" : ""}`} />{online() === null ? "connecting" : online() ? "connected" : "offline"}</span>
          <label class="pill"><input type="checkbox" checked={multi()} onChange={(e) => setMulti(e.currentTarget.checked)} /> keep conversation</label>
        </div>
        <div class="msgs" ref={box}>
          <Show when={msgs().length === 0}>
            <p style={{ color: "var(--muted)" }}>Every word below is computed on the Zybo: the FPGA streams the weights, the ARM does the rest. This page only turns text into token ids.</p>
            <div class="suggest"><For each={SUGGEST}>{(s) => <button onClick={() => ask(s)}>{s}</button>}</For></div>
          </Show>
          <For each={msgs()}>{(m) => (
            <div class={`msg ${m.role}`}>{m.text || (m.role === "bot" ? "…" : "")}<Show when={m.meta}><span class="meta">{m.meta}</span></Show></div>
          )}</For>
        </div>
        <form class="composer" onSubmit={(e) => { e.preventDefault(); ask(); }}>
          <input value={input()} onInput={(e) => setInput(e.currentTarget.value)} placeholder="Ask a short factual question" />
          <button disabled={busy() || !online()}>Ask</button>
        </form>
      </section>

      <section style={{ display: "grid", gap: "16px", "align-content": "start" }}>
        <div class="panel">
          <h3>What the hardware is doing</h3>
          <Arch pl={phase() !== "idle"} arm={phase() !== "idle"}
                label={phase() === "idle" ? "idle" : phase() === "prefill" ? "reading the prompt: one pass over the weights per prompt token" : "generating: one pass over 490 MB of weights per token"} />
        </div>
        <div class="panel">
          <h3>Time per token</h3>
          <Show when={tokMs().length} fallback={<p style={{ color: "var(--muted)", margin: 0 }}>Ask something to see per-token timing.</p>}>
            <svg viewBox="0 0 600 90" role="img" aria-label="Milliseconds per generated token">
              <For each={tokMs()}>{(ms, i) => {
                const h = Math.min(80, (ms / 400) * 80);
                return <rect x={i() * 10} y={86 - h} width="7" height={h} rx="2" fill="var(--accent)" />;
              }}</For>
            </svg>
            <div class="row">last {tokMs()[tokMs().length - 1]} ms · bars scaled to 400 ms</div>
          </Show>
        </div>
        <div class="panel">
          <h3>Where one token's time goes</h3>
          <Show when={profEntries().length} fallback={<p style={{ color: "var(--muted)", margin: 0 }}>The board reports a profile after each answer.</p>}>
            <div class="bars">
              <For each={profEntries()}>{([k, v]) => (
                <div class="bar">
                  <span>{k.replace("_", " ")}</span>
                  <div class="track"><div class="fill" style={{ width: `${(v / profTotal()) * 100}%`, background: COLORS[k] ?? "var(--muted)" }} /></div>
                  <span class="v">{(v / 1000).toFixed(1)} ms</span>
                </div>
              )}</For>
            </div>
          </Show>
        </div>
        <Show when={cand()}>
          <div class="panel">
            <h3>Top candidates for the last token</h3>
            <div class="cands">
              <For each={cand()!.texts}>{(t, i) => {
                const l = cand()!.logits;
                const p = Math.exp(l[i()] - l[0]);
                return (
                  <div class="cand">
                    <div class="w"><i style={{ transform: `scaleX(${p})` }} /><span>{JSON.stringify(t)}</span></div>
                    <span class="v" style={{ "text-align": "right", color: "var(--muted)" }}>{l[i()].toFixed(2)}</span>
                  </div>
                );
              }}</For>
            </div>
          </div>
        </Show>
      </section>
    </div>
  );
}
