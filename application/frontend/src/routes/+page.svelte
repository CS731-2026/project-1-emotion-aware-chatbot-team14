<script lang="ts">
  import type { PageData } from "./$types";
  import Button from "$lib/components/Button.svelte";

  let { data }: { data: PageData } = $props();

  let clicked = $state(false);
</script>

<h1>Hello from SvelteKit</h1>
<p>Server says: {data.message}</p>
<p>Timestamp: {data.timestamp}</p>

<Button label={clicked ? "Clicked!" : "Click me"} onclick={() => (clicked = !clicked)} />
