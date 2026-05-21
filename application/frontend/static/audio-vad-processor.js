// AudioWorklet processor that forwards every input frame (128 mono samples)
// back to the main thread via the message port. Doing analysis on the main
// thread keeps this file simple and matches MediaRecorder's old data path.
class VadProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const channel = input[0];
    if (!channel || channel.length === 0) return true;
    // .slice() because the engine reuses the channel buffer.
    this.port.postMessage(channel.slice());
    return true;
  }
}
registerProcessor("vad-processor", VadProcessor);
