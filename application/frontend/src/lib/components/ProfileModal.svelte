<script lang="ts">
  import { api, type Profile } from "$lib/api";

  let { onSelected }: { onSelected: (profile: Profile) => void } = $props();

  let profiles = $state<Profile[]>([]);
  let newName = $state("");
  let loading = $state(true);
  let error = $state("");

  $effect(() => {
    api.getProfiles().then((p) => {
      profiles = p;
      loading = false;
    });
  });

  async function select(profile: Profile) {
    await api.selectProfile(profile.id);
    onSelected(profile);
  }

  async function create() {
    const name = newName.trim();
    if (!name) return;
    error = "";
    try {
      const profile = await api.createProfile(name);
      await api.selectProfile(profile.id);
      onSelected(profile);
    } catch (e) {
      error = "Failed to create profile";
    }
  }
</script>

<div class="backdrop">
  <div class="modal">
    <h2>Who are you?</h2>
    <p class="sub">Select a profile to continue, or create a new one.</p>

    {#if loading}
      <p class="loading">Loading…</p>
    {:else}
      {#if profiles.length > 0}
        <ul class="profile-list">
          {#each profiles as p (p.id)}
            <li>
              <button onclick={() => select(p)}>{p.name}</button>
            </li>
          {/each}
        </ul>
        <hr />
      {/if}

      <form onsubmit={(e) => { e.preventDefault(); create(); }}>
        <input
          type="text"
          bind:value={newName}
          placeholder="New profile name"
          maxlength="40"
        />
        <button type="submit" class="create-btn" disabled={!newName.trim()}>Create & select</button>
      </form>

      {#if error}
        <p class="error">{error}</p>
      {/if}
    {/if}
  </div>
</div>

<style>
  .backdrop {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 100;
  }

  .modal {
    background: hsl(220, 15%, 14%);
    border: 1px solid var(--color-border);
    border-radius: calc(var(--radius) * 1.5);
    padding: 2rem;
    width: min(380px, 90vw);
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  h2 {
    font-size: 1.3rem;
  }

  .sub {
    font-size: 0.85rem;
    color: var(--color-text-muted);
  }

  .loading {
    color: var(--color-text-muted);
  }

  .profile-list {
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }

  .profile-list button {
    width: 100%;
    text-align: left;
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius);
    color: var(--color-text);
    cursor: pointer;
    font-size: 0.95rem;
    padding: 0.6rem 0.9rem;
    transition: background 0.15s;
  }

  .profile-list button:hover {
    background: var(--color-surface-hover);
  }

  hr {
    border: none;
    border-top: 1px solid var(--color-border);
  }

  form {
    display: flex;
    gap: 0.5rem;
  }

  input {
    flex: 1;
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius);
    color: var(--color-text);
    font-size: 0.95rem;
    padding: 0.55rem 0.8rem;
    outline: none;
    font-family: var(--font);
  }

  input:focus {
    border-color: rgba(255, 255, 255, 0.3);
  }

  .create-btn {
    background: rgba(255, 255, 255, 0.15);
    border: 1px solid var(--color-border);
    border-radius: var(--radius);
    color: var(--color-text);
    cursor: pointer;
    font-size: 0.9rem;
    padding: 0.55rem 0.9rem;
    white-space: nowrap;
    transition: background 0.15s;
  }

  .create-btn:hover:not(:disabled) {
    background: rgba(255, 255, 255, 0.22);
  }

  .create-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .error {
    color: hsl(0, 70%, 65%);
    font-size: 0.85rem;
  }
</style>
