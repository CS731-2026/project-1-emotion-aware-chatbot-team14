<script lang="ts">
  let { isSpeaking }: { isSpeaking: boolean } = $props();
</script>

<div class="circle-wrap">
  <div class="circle" class:speaking={isSpeaking}>
    <span class="ring ring1"></span>
    <span class="ring ring2"></span>
  </div>
</div>

<style>
  .circle-wrap {
    display: flex;
    justify-content: center;
    padding: 1.5rem 0 0.8rem;
  }

  .circle {
    position: relative;
    width: 74px;
    height: 74px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.88);
    flex-shrink: 0;
    box-shadow: 0 0 35px rgba(255, 255, 255, 0.12);
  }

  .ring {
    position: absolute;
    inset: 0;
    border-radius: 50%;
    border: 2px solid rgba(255, 255, 255, 0.6);
    opacity: 0;
  }

  .speaking .ring1 {
    animation: pulse 1.6s ease-out infinite;
  }

  .speaking .ring2 {
    animation: pulse 1.6s ease-out 0.5s infinite;
  }

  @keyframes pulse {
    0% { transform: scale(1); opacity: 0.68; }
    100% { transform: scale(2.5); opacity: 0; }
  }
</style>
