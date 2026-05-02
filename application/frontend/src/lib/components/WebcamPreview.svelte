<script lang="ts">
  let { stream }: { stream: MediaStream | null } = $props();

  let videoEl = $state<HTMLVideoElement | undefined>(undefined);

  $effect(() => {
    if (videoEl && stream) {
      videoEl.srcObject = stream;
    }
  });
</script>

<div class="preview">
  {#if stream}
    <video bind:this={videoEl} autoplay muted playsinline></video>
  {:else}
    <div class="placeholder">📷</div>
  {/if}
</div>

<style>
  .preview {
    position: fixed;
    bottom: 5.5rem;
    right: 1rem;
    width: 140px;
    height: 105px;
    border-radius: var(--radius);
    overflow: hidden;
    border: 1px solid var(--color-border);
    background: rgba(0, 0, 0, 0.4);
  }

  video {
    width: 100%;
    height: 100%;
    object-fit: cover;
    transform: scaleX(-1);
  }

  .placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    font-size: 2rem;
    opacity: 0.4;
  }
</style>
