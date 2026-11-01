// Measured numbers used by the write-up. Every value here comes from a board log in
// docs/results/ (or the profile the firmware prints); update them together.

export const model = {
  name: "BitNet b1.58 2B-4T",
  params: 2.41e9, // incl. embedding
  layers: 30,
  hidden: 2560,
  ffn: 6912,
  heads: 20,
  kvHeads: 5,
  vocab: 128256,
};

export const board = {
  name: "Digilent Zybo Z7-10",
  chip: "Zynq-7010 (2 x Cortex-A9 @ 667 MHz + Artix-7 fabric)",
  ddr: "1 GB DDR3L, 32-bit @ 1066 MT/s",
  ddrPeakGBs: 4.27,
  plMHz: 142.857,
};

// Bytes the engine must read per generated token, by design generation.
export const bytesPerToken = [
  { label: "v1: 2-bit layers + int8 head", layers: 521, head: 328 },
  { label: "v1 + coarse ternary head", layers: 521, head: 82 },
  { label: "v2: 1.6-bit layers + 1.6-bit coarse head", layers: 424, head: 66 },
];

// Decode speed through the project (ms per generated token, short answer, measured).
export const milestones = [
  { step: "Original design (PC drives every matmul over UART)", ms: 519000 },
  { step: "Streaming engine, weights in DDR, int8 head", ms: 293 },
  { step: "NEON kernels + coarse ternary LM head", ms: 195 },
  { step: "1.6-bit engine, overlap, 2 cores, speculation (100 MHz, USB power)", ms: 121 },
];

// One pass of the current firmware (us), France prompt, engine at 100 MHz (USB power).
// The engine stream includes the A9 epilogue work that runs while it streams.
export const profile: Record<string, number> = {
  "engine stream": 153934,
  attention: 6799,
  quantize: 4489,
  rope: 1615,
  "norm stats": 1479,
  "head rescoring": 547,
  "x copy": 541,
  embed: 112,
};

export const answers = [
  { q: "What is the capital of France?", a: "The capital of France is Paris." },
  { q: "How many legs does a spider have?", a: "Spiders have eight legs. This characteristic is common to most species of spiders." },
  { q: "What is 12 times 7?", a: "12 times 7 equals 84." },
  { q: "Name a planet in our solar system that has rings.", a: "Saturn is the planet in our solar system that has rings." },
];
