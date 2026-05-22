<script lang="ts">
  import type { AssistantPhase } from "$lib/conversation/uiState";

  let {
    phase,
    pulse = 0,
    compact = false,
    mismatch = false,
  }: {
    phase: AssistantPhase;
    pulse?: number;
    compact?: boolean;
    mismatch?: boolean;
  } = $props();

  const clampedPulse = $derived(Math.max(0, Math.min(pulse, 1)));
  const coreScale = $derived(1 + clampedPulse * 0.16);
  const haloScale = $derived(1 + clampedPulse * 0.32);
</script>

<div class="circle-wrap" class:compact>
  <div
    class="visualiser {phase}"
    class:mismatch
    style={`--core-scale:${coreScale}; --halo-scale:${haloScale}; --pulse-opacity:${0.28 + clampedPulse * 0.52};`}
    aria-hidden="true"
  >
    <span class="halo"></span>
    <span class="core"></span>
    <span class="ring ring-one"></span>
    <span class="ring ring-two"></span>
  </div>
</div>

<style>
  .circle-wrap {
    display: flex;
    justify-content: center;
    padding: 1rem 0;
    transition: transform 320ms cubic-bezier(0.22, 1, 0.36, 1),
                opacity 320ms cubic-bezier(0.22, 1, 0.36, 1);
    transform-origin: center;
  }

  /* Compact: shrink + dock to top-right as a "still listening" marker while
     an overlay check-in is active. */
  .circle-wrap.compact {
    transform: scale(0.4) translate(160%, -110%);
    opacity: 0.75;
  }

  .visualiser {
    --core-scale: 1;
    --halo-scale: 1;
    --pulse-opacity: 0.4;
    position: relative;
    width: clamp(220px, 28vw, 320px);
    aspect-ratio: 1;
    display: grid;
    place-items: center;
    filter: drop-shadow(0 18px 44px rgba(0, 0, 0, 0.28));
  }

  .halo,
  .core,
  .ring {
    position: absolute;
    border-radius: 50%;
  }

  .halo {
    inset: 8%;
    background:
      radial-gradient(circle at 30% 30%, rgba(255, 255, 255, 0.72), rgba(255, 255, 255, 0.05) 60%, transparent 72%),
      radial-gradient(circle at 70% 70%, rgba(255, 255, 255, 0.3), transparent 65%);
    opacity: var(--pulse-opacity);
    transform: scale(var(--halo-scale));
    transition: transform 140ms linear, opacity 140ms linear;
  }

  .core {
    inset: 22%;
    background:
      radial-gradient(circle at 35% 35%, rgba(255, 255, 255, 0.98), rgba(255, 255, 255, 0.86) 38%, rgba(255, 255, 255, 0.22) 72%, rgba(255, 255, 255, 0.08) 100%);
    transform: scale(var(--core-scale));
    transition: transform 120ms linear;
    box-shadow:
      inset 0 0 32px rgba(255, 255, 255, 0.45),
      0 0 60px rgba(255, 255, 255, 0.18);
  }

  .ring {
    inset: 6%;
    border: 1px solid rgba(255, 255, 255, 0.28);
    opacity: 0;
  }

  .speaking .ring-one {
    animation: orbit 1.35s ease-out infinite;
  }

  .speaking .ring-two {
    animation: orbit 1.35s ease-out 0.45s infinite;
  }

  .listening .halo,
  .ready .halo {
    animation: breathe 3.2s ease-in-out infinite;
  }

  .thinking .core {
    animation: think 1.8s ease-in-out infinite;
  }

  .offline .core,
  .needs-profile .core {
    opacity: 0.8;
  }

  @keyframes orbit {
    0% {
      transform: scale(0.92);
      opacity: 0.68;
    }
    100% {
      transform: scale(1.24);
      opacity: 0;
    }
  }

  @keyframes breathe {
    0%, 100% {
      transform: scale(1);
      opacity: 0.34;
    }
    50% {
      transform: scale(1.05);
      opacity: 0.54;
    }
  }

  @keyframes think {
    0%, 100% {
      transform: scale(1);
    }
    50% {
      transform: scale(1.03);
    }
  }

  .visualiser.mismatch .core {
    background:
      radial-gradient(circle at 35% 35%, rgba(255,120,120,0.98), rgba(220,60,60,0.86) 38%, rgba(180,30,30,0.22) 72%, rgba(140,0,0,0.08) 100%);
    box-shadow:
      inset 0 0 32px rgba(255,80,80,0.45),
      0 0 60px rgba(220,50,50,0.28);
    transition: background 600ms ease, box-shadow 600ms ease;
  }
  .visualiser.mismatch .halo {
    background:
      radial-gradient(circle at 30% 30%, rgba(255,160,160,0.72), rgba(220,80,80,0.05) 60%, transparent 72%),
      radial-gradient(circle at 70% 70%, rgba(200,60,60,0.3), transparent 65%);
  }
</style>
