<script lang="ts">
  import { slide } from "svelte/transition";
  import { cubicOut } from "svelte/easing";

  let { text }: { text: string } = $props();
</script>

<div class="alert" transition:slide={{ duration: 220, easing: cubicOut }}>
  <span class="icon" aria-hidden="true">!</span>
  <p>{text}</p>
</div>

<style>
  .alert {
    display: flex;
    gap: 0.75rem;
    align-items: flex-start;
    padding: 0.85rem 1rem;
    border-radius: 12px;
    border: 1px solid rgba(217, 119, 6, 0.45);
    background: rgba(254, 243, 199, 0.5);
    color: #78350f;
    font-size: 0.95rem;
    line-height: 1.5;
  }
  .icon {
    width: 1.25rem;
    height: 1.25rem;
    border-radius: 50%;
    border: 1.5px solid currentColor;
    display: grid;
    place-items: center;
    font-weight: 700;
    font-size: 0.8rem;
    flex-shrink: 0;
    margin-top: 0.1rem;
  }
  p {
    margin: 0;
  }
</style>
