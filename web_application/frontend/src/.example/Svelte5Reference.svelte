<!--
  SVELTE 5 QUICK REFERENCE
  This file is intentionally outside src/routes/ — it is never built or served.
  Use it as a local cheat sheet for Svelte 5 rune-based syntax.
-->

<script lang="ts">
  // ── $state ───────────────────────────────────────────────────────────────
  // Reactive primitive. Replaces `let x` with reactivity in Svelte 4.
  let count = $state(0);
  let name = $state("world");

  // $state with objects — deep reactivity on class instances / plain objects
  let user = $state({ name: "Alice", age: 30 });

  // ── $derived ─────────────────────────────────────────────────────────────
  // Computed value. Re-evaluates whenever its dependencies change.
  let doubled = $derived(count * 2);
  let greeting = $derived(`Hello, ${name}!`);

  // $derived.by — for multi-line derivations
  let summary = $derived.by(() => {
    const base = count * 2;
    return base > 10 ? "big" : "small";
  });

  // ── $effect ───────────────────────────────────────────────────────────────
  // Runs after the component mounts and re-runs when dependencies change.
  // Replaces `$: { ... }` side-effect blocks and onMount for reactive side effects.
  $effect(() => {
    console.log("count changed:", count);
    // Return a cleanup function (optional)
    return () => console.log("cleanup");
  });

  // $effect.pre — runs before the DOM updates (rarely needed)
  $effect.pre(() => {
    console.log("before DOM update, count is:", count);
  });

  // ── $props ────────────────────────────────────────────────────────────────
  // Replaces `export let`. Only valid at the top level of a component.
  // Shown here as a comment — $props() is used in child components, not here.
  //
  //   let { label, onclick, variant = "primary" }: Props = $props();

  // ── $bindable ────────────────────────────────────────────────────────────
  // Marks a prop as two-way bindable from the parent.
  // Used inside a child component:
  //
  //   let { value = $bindable("") }: { value: string } = $props();
  //
  // Parent binds with:  <Child bind:value={myVar} />

  // ── $inspect ─────────────────────────────────────────────────────────────
  // Dev-only: logs value + any reactive updates. Stripped from production.
  $inspect(count, user);

  // ── Snippets ──────────────────────────────────────────────────────────────
  // Reusable template fragments defined inline. Replaces named slots.
  // Defined with {#snippet} and called with {@render}.

  // ── Event handlers ────────────────────────────────────────────────────────
  // onclick / oninput / onkeydown etc. are plain HTML event attributes now.
  // No more `on:click` directive.
  function increment() {
    count++;
  }
</script>

<!-- ── $state & events ─────────────────────────────────────────────────── -->
<section>
  <h2>$state + events</h2>
  <button onclick={increment}>Clicked {count} times</button>
  <p>Doubled: {doubled} ({summary})</p>
</section>

<!-- ── Inline handler ──────────────────────────────────────────────────── -->
<section>
  <h2>Inline handler</h2>
  <button onclick={() => (name = name === "world" ? "Svelte" : "world")}>
    Toggle name
  </button>
  <p>{greeting}</p>
</section>

<!-- ── $state object (deep reactive) ──────────────────────────────────── -->
<section>
  <h2>$state object</h2>
  <input bind:value={user.name} />
  <p>User: {user.name}, age {user.age}</p>
</section>

<!-- ── {#snippet} + {@render} ─────────────────────────────────────────── -->
<section>
  <h2>Snippets</h2>

  {#snippet badge(text: string, color: string)}
    <span style="background:{color}; padding:2px 8px; border-radius:9999px">
      {text}
    </span>
  {/snippet}

  {@render badge("active", "#4ade80")}
  {@render badge("pending", "#facc15")}
</section>

<!-- ── {#each} ─────────────────────────────────────────────────────────── -->
<section>
  <h2>{`{#each}`}</h2>
  {#each ["apple", "banana", "cherry"] as fruit, i (fruit)}
    <p>{i + 1}. {fruit}</p>
  {/each}
</section>

<!-- ── {#if} / {:else if} / {:else} ───────────────────────────────────── -->
<section>
  <h2>{`{#if}`}</h2>
  {#if count > 5}
    <p>Count is greater than 5</p>
  {:else if count > 0}
    <p>Count is positive</p>
  {:else}
    <p>Count is zero</p>
  {/if}
</section>

<!-- ── {#await} ────────────────────────────────────────────────────────── -->
<section>
  <h2>{`{#await}`}</h2>
  {#await Promise.resolve("loaded!")}
    <p>Loading...</p>
  {:then value}
    <p>Result: {value}</p>
  {:catch error}
    <p>Error: {error.message}</p>
  {/await}
</section>

<!-- ── bind: ────────────────────────────────────────────────────────────── -->
<section>
  <h2>bind:</h2>
  <input bind:value={name} placeholder="type a name" />
  <p>Bound value: {name}</p>
</section>
