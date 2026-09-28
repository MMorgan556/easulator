// Web Worker that owns the ONNX Runtime session (wasm backend, runs fully on-device).
import * as ort from "../vendor/onnxruntime-web/ort.wasm.min.mjs";

ort.env.wasm.wasmPaths = new URL("../vendor/onnxruntime-web/", import.meta.url).href;
// GitHub Pages can't send the cross-origin isolation headers threads need.
ort.env.wasm.numThreads = 1;

let session = null;

self.onmessage = async ({ data }) => {
  const { id } = data;
  try {
    if (data.type === "load") {
      session = await ort.InferenceSession.create(data.url, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
      self.postMessage({ id, inputNames: session.inputNames, outputNames: session.outputNames });
    } else if (data.type === "run") {
      if (!session) throw new Error("Model is not loaded");
      const input = new ort.Tensor("float32", data.input, [1, 3, data.size, data.size]);
      const outputs = await session.run({ [session.inputNames[0]]: input });
      const output = outputs[session.outputNames[0]];
      const copy = new Float32Array(output.data);
      self.postMessage({ id, data: copy, dims: output.dims }, [copy.buffer]);
    }
  } catch (err) {
    self.postMessage({ id, error: err?.message || String(err) });
  }
};
