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
    <div class="placeholder">Camera off</div>
  {/if}
</div>

<style>
  .preview {
    position: fixed;
    bottom: 1rem;
    right: 1rem;
    width: 168px;
    height: 126px;
    border-radius: var(--radius);
    overflow: hidden;
    border: 1px solid var(--color-border);
    background: rgba(0, 0, 0, 0.45);
    z-index: 5;
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
    font-size: 0.9rem;
    color: var(--color-text-muted);
  }
</style>
