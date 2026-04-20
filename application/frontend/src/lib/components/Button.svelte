<script lang="ts">
  type Props = {
    label: string;
    onclick?: () => void;
    variant?: "primary" | "secondary";
  };

  let { label, onclick, variant = "primary" }: Props = $props();
</script>

<button class={variant} {onclick}>
  {label}
</button>

<style>
  button {
    padding: 0.5rem 1rem;
    border: none;
    border-radius: 4px;
    cursor: pointer;
  }
  .primary {
    background: #4f46e5;
    color: white;
  }
  .secondary {
    background: #e5e7eb;
    color: #111;
  }
</style>
