<script lang="ts">
  import { slide } from "svelte/transition";
  import { cubicOut } from "svelte/easing";

  let {
    speaker = "Your GP's assistant",
    text,
  }: { speaker?: string; text: string } = $props();
</script>

<div class="bubble" transition:slide={{ duration: 220, easing: cubicOut }}>
  <p class="speaker">{speaker}</p>
  <p class="text">{text}</p>
</div>

<style>
  .bubble {
    padding: 0.95rem 1.1rem;
    border-radius: 12px;
    background: rgba(199, 210, 254, 0.4);
    color: #1e1b4b;
  }
  .speaker {
    margin: 0 0 0.4rem;
    font-weight: 600;
    color: #4338ca;
    font-size: 0.92rem;
  }
  .text {
    margin: 0;
    line-height: 1.55;
    font-size: 0.98rem;
  }
</style>
