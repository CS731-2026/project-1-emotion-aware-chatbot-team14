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
    }).catch(() => {
      error = "Could not load profiles from backend";
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
    } catch {
      error = "Failed to create profile";
    }
  }
</script>

<div class="backdrop">
  <div class="modal">
    <h2>Who are you?</h2>
    <p class="sub">Select a profile to continue, or create a new one.</p>

    {#if loading}
      <p class="loading">Loading...</p>
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
    border-radius: calc(var(--radius) * 1.1);
    padding: 2rem;
    width: min(380px, 90vw);
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  h2 {
    font-size: 1.3rem;
  }

  .sub,
  .loading {
    font-size: 0.9rem;
    color: var(--color-text-muted);
  }

  .profile-list {
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
  }

  .profile-list button,
  .create-btn,
  input {
    width: 100%;
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius);
    color: var(--color-text);
    font-size: 0.95rem;
    padding: 0.65rem 0.85rem;
  }

  .profile-list button {
    text-align: left;
    cursor: pointer;
  }

  .profile-list button:hover,
  .create-btn:hover:not(:disabled) {
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
  }

  .create-btn {
    width: auto;
    white-space: nowrap;
    cursor: pointer;
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
