<script lang="ts">
  import { browser } from "$app/environment";

  const DASHBOARD_PASSWORD = "testpwd";
  const UNLOCK_KEY = "dashboard:unlocked";

  let unlocked = $state(
    browser ? localStorage.getItem(UNLOCK_KEY) === "1" : false
  );
  let passwordInput = $state("");
  let passwordError = $state(false);

  function tryUnlock() {
    if (passwordInput === DASHBOARD_PASSWORD) {
      unlocked = true;
      passwordError = false;
      if (browser) localStorage.setItem(UNLOCK_KEY, "1");
    } else {
      passwordError = true;
      passwordInput = "";
    }
  }

  function lock() {
    unlocked = false;
    if (browser) localStorage.removeItem(UNLOCK_KEY);
  }
</script>

{#if !unlocked}
  <!-- ── Password gate ─────────────────────────────────────────────── -->
  <div class="gate-wrap">
    <div class="gate-card">
      <div class="gate-logo">
        <div class="logo-dot"></div>
        <span class="logo-text">EmpathyCheck</span>
      </div>
      <h1 class="gate-heading">GP Dashboard</h1>
      <p class="gate-sub">Enter your practice password to continue.</p>
      <form onsubmit={(e) => { e.preventDefault(); tryUnlock(); }}>
        <input
          class="gate-input"
          class:gate-input-error={passwordError}
          type="password"
          placeholder="Practice password"
          bind:value={passwordInput}
          autocomplete="current-password"
        />
        {#if passwordError}
          <p class="gate-error">Incorrect password — please try again.</p>
        {/if}
        <button class="gate-btn" type="submit">Sign in →</button>
      </form>
      <a class="gate-back" href="/">← Back to app</a>
    </div>
  </div>

{:else}
  <!-- ── Dashboard ──────────────────────────────────────────────────── -->
  <div class="db">
    <!-- Topbar -->
    <div class="topbar">
      <div class="topbar-left">
        <div class="logo-dot"></div>
        <span class="logo-text">EmpathyCheck</span>
        <span class="practice-pill">Riverside Medical Practice</span>
      </div>
      <div class="topbar-right">
        <span class="month-badge">📅 May 2026</span>
        <div class="avatar">DK</div>
        <button class="lock-btn" onclick={lock}>Sign out</button>
      </div>
    </div>

    <!-- Nav -->
    <nav class="nav">
      <div class="nav-item active">Overview</div>
      <div class="nav-item">Responses</div>
      <div class="nav-item">Mismatch alerts</div>
      <div class="nav-item">Feedback links</div>
      <div class="nav-item">Settings</div>
    </nav>

    <div class="main">

      <!-- KPI metrics -->
      <div class="metrics">
        <div class="metric">
          <div class="metric-label">Responses this month</div>
          <div class="metric-val">148</div>
          <div class="metric-sub up">↑ +12 vs last month</div>
        </div>
        <div class="metric">
          <div class="metric-label">Emotion mismatches</div>
          <div class="metric-val">31</div>
          <div class="metric-sub warn">21% of responses</div>
        </div>
        <div class="metric">
          <div class="metric-label">Questions asked</div>
          <div class="metric-val">87</div>
          <div class="metric-sub up">↑ 59% ask rate</div>
        </div>
        <div class="metric">
          <div class="metric-label">Avg comfort score</div>
          <div class="metric-val">7.4<span class="metric-denom">/10</span></div>
          <div class="metric-sub up">↑ +0.6 this month</div>
        </div>
      </div>

      <!-- Row 2: Mismatch alerts + Comfort chart -->
      <div class="row2">
        <div class="card">
          <div class="card-header">
            <span class="card-title">⚠️ Recent mismatch alerts</span>
            <span class="card-action">View all</span>
          </div>
          <div class="mismatch-list">
            <div class="mismatch-item">
              <div class="mismatch-icon warn">😟</div>
              <div class="mismatch-text">
                <div class="mismatch-q">AI use during appointment <span class="tag tag-amber">mismatch</span></div>
                <div class="mismatch-detail">Said "I feel fine" — emotion signal: anxious (score 0.28). LLM explanation triggered.</div>
              </div>
              <div class="mismatch-time">2 hrs ago</div>
            </div>
            <div class="mismatch-item">
              <div class="mismatch-icon warn">😟</div>
              <div class="mismatch-text">
                <div class="mismatch-q">Data privacy concern <span class="tag tag-amber">mismatch</span></div>
                <div class="mismatch-detail">Said "Very comfortable" — dwell time 14s, re-selected twice. Score: 0.31.</div>
              </div>
              <div class="mismatch-time">5 hrs ago</div>
            </div>
            <div class="mismatch-item">
              <div class="mismatch-icon ok">😊</div>
              <div class="mismatch-text">
                <div class="mismatch-q">Doctor listened to me <span class="tag tag-teal">matched</span></div>
                <div class="mismatch-detail">Said "Yes, completely" — emotion signal confirmed positive (score 0.84).</div>
              </div>
              <div class="mismatch-time">Yesterday</div>
            </div>
            <div class="mismatch-item">
              <div class="mismatch-icon warn">😟</div>
              <div class="mismatch-text">
                <div class="mismatch-q">AI making decisions for me <span class="tag tag-amber">mismatch</span></div>
                <div class="mismatch-detail">Said "Mostly fine" — hesitation detected, follow-up question submitted.</div>
              </div>
              <div class="mismatch-time">Yesterday</div>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <span class="card-title">Comfort with AI — breakdown</span>
          </div>
          <div class="bar-group">
            <div class="bar-row">
              <span class="bar-label">Very comfortable</span>
              <div class="bar-track"><div class="bar-fill" style="width:38%; background:#5DCAA5;"></div></div>
              <span class="bar-pct">38%</span>
            </div>
            <div class="bar-row">
              <span class="bar-label">Mostly fine</span>
              <div class="bar-track"><div class="bar-fill" style="width:29%; background:#9FE1CB;"></div></div>
              <span class="bar-pct">29%</span>
            </div>
            <div class="bar-row">
              <span class="bar-label">A bit unsure</span>
              <div class="bar-track"><div class="bar-fill" style="width:21%; background:#FAC775;"></div></div>
              <span class="bar-pct">21%</span>
            </div>
            <div class="bar-row">
              <span class="bar-label">Worried</span>
              <div class="bar-track"><div class="bar-fill" style="width:12%; background:#F0997B;"></div></div>
              <span class="bar-pct">12%</span>
            </div>
          </div>
          <div class="trend-section">
            <p class="section-divider">Weekly response trend</p>
            <div class="trend-bars">
              <div class="trend-bar-wrap"><div class="trend-bar" style="height:40px; background:#CECBF6;"></div><span class="trend-label">W1</span></div>
              <div class="trend-bar-wrap"><div class="trend-bar" style="height:52px; background:#AFA9EC;"></div><span class="trend-label">W2</span></div>
              <div class="trend-bar-wrap"><div class="trend-bar" style="height:48px; background:#AFA9EC;"></div><span class="trend-label">W3</span></div>
              <div class="trend-bar-wrap"><div class="trend-bar" style="height:68px; background:#534AB7;"></div><span class="trend-label">W4</span></div>
            </div>
          </div>
        </div>
      </div>

      <!-- Row 3: Feedback links + Top questions -->
      <div class="row3">
        <div class="card">
          <div class="card-header">
            <span class="card-title">🔗 Feedback links</span>
            <button class="generate-btn">＋ New link</button>
          </div>
          <div class="link-card">
            <div>
              <div class="link-meta">Email link — Dr. Khan's patients</div>
              <div class="link-url">empathy.check/r/dk-may26</div>
            </div>
            <div class="link-actions">
              <button class="link-btn">📋 Copy</button>
              <button class="link-btn">📊 64</button>
            </div>
          </div>
          <div class="link-card">
            <div>
              <div class="link-meta">Kiosk — waiting room device</div>
              <div class="link-url">empathy.check/r/kiosk-rv</div>
            </div>
            <div class="link-actions">
              <button class="link-btn">📋 Copy</button>
              <button class="link-btn">📊 84</button>
            </div>
          </div>
          <p class="kiosk-note">🖥 Kiosk auto-resets after each session</p>
        </div>

        <div class="card">
          <div class="card-header">
            <span class="card-title">💬 Top patient questions</span>
            <span class="card-action">Export</span>
          </div>
          <div class="question-item">
            <div class="q-count">Asked 34 times</div>
            <div class="q-text">"Will my doctor see this feedback?" <span class="tag tag-purple">privacy</span></div>
          </div>
          <div class="question-item">
            <div class="q-count">Asked 27 times</div>
            <div class="q-text">"What did the AI actually do?" <span class="tag tag-amber">AI role</span></div>
          </div>
          <div class="question-item">
            <div class="q-count">Asked 18 times</div>
            <div class="q-text">"Is my data kept safe?" <span class="tag tag-purple">privacy</span></div>
          </div>
          <div class="question-item">
            <div class="q-count">Asked 8 times</div>
            <div class="q-text">"Can I opt out of AI next time?" <span class="tag tag-teal">preference</span></div>
          </div>
        </div>
      </div>

    </div><!-- /main -->
  </div><!-- /db -->
{/if}

<style>
  /* ── Gate ───────────────────────────────────────────────────────── */
  .gate-wrap {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #f0f0f5;
    padding: 1.5rem;
  }
  .gate-card {
    background: #fff;
    border-radius: 20px;
    padding: 2.5rem 2rem;
    width: min(400px, 100%);
    display: flex;
    flex-direction: column;
    gap: 1rem;
    box-shadow: 0 8px 40px rgba(0,0,0,0.12);
  }
  .gate-logo {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .gate-heading {
    font-size: 1.6rem;
    font-weight: 800;
    color: #111;
    margin: 0;
  }
  .gate-sub {
    font-size: 0.9rem;
    color: #666;
    margin: 0;
  }
  .gate-input {
    width: 100%;
    padding: 0.75rem 1rem;
    border: 1px solid #ddd;
    border-radius: 10px;
    font-size: 0.95rem;
    outline: none;
    transition: border-color 150ms;
  }
  .gate-input:focus { border-color: #534AB7; }
  .gate-input-error { border-color: #e55; }
  .gate-error {
    font-size: 0.82rem;
    color: #c33;
    margin: -0.4rem 0 0;
  }
  .gate-btn {
    width: 100%;
    padding: 0.85rem;
    background: #1a1a2e;
    color: #fff;
    border: none;
    border-radius: 10px;
    font-size: 0.95rem;
    font-weight: 700;
    cursor: pointer;
    transition: background 140ms;
  }
  .gate-btn:hover { background: #2d2d50; }
  .gate-back {
    text-align: center;
    font-size: 0.85rem;
    color: #888;
    text-decoration: none;
  }
  .gate-back:hover { color: #333; }

  /* ── Dashboard shell ─────────────────────────────────────────────── */
  .db {
    min-height: 100vh;
    background: #f4f4f8;
    color: #1a1a2e;
    font-family: system-ui, sans-serif;
  }

  /* Topbar */
  .topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.85rem 1.25rem;
    background: #fff;
    border-bottom: 1px solid #e5e5ea;
    gap: 1rem;
    flex-wrap: wrap;
  }
  .topbar-left { display: flex; align-items: center; gap: 0.6rem; }
  .topbar-right { display: flex; align-items: center; gap: 0.65rem; }

  .logo-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #534AB7;
    flex-shrink: 0;
  }
  .logo-text { font-size: 0.95rem; font-weight: 600; color: #1a1a2e; }
  .practice-pill {
    font-size: 0.75rem;
    color: #3C3489;
    background: #EEEDFE;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    border: 1px solid #AFA9EC;
  }
  .month-badge {
    font-size: 0.8rem;
    color: #555;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 0.3rem 0.7rem;
    background: #fff;
  }
  .avatar {
    width: 32px; height: 32px;
    border-radius: 50%;
    background: #EEEDFE;
    color: #3C3489;
    font-size: 0.75rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .lock-btn {
    font-size: 0.78rem;
    padding: 0.3rem 0.7rem;
    border: 1px solid #ddd;
    border-radius: 8px;
    background: #fff;
    color: #666;
    cursor: pointer;
  }
  .lock-btn:hover { background: #f5f5f7; }

  /* Nav */
  .nav {
    display: flex;
    gap: 2px;
    padding: 0 1.25rem;
    background: #fff;
    border-bottom: 1px solid #e5e5ea;
    overflow-x: auto;
  }
  .nav-item {
    font-size: 0.82rem;
    padding: 0.6rem 0.9rem 0.65rem;
    color: #888;
    cursor: default;
    border-bottom: 2px solid transparent;
    white-space: nowrap;
  }
  .nav-item.active {
    color: #534AB7;
    border-bottom-color: #534AB7;
    font-weight: 600;
  }

  /* Main content */
  .main { padding: 1.25rem; }

  /* Metrics */
  .metrics {
    display: grid;
    grid-template-columns: repeat(4, minmax(0,1fr));
    gap: 0.75rem;
    margin-bottom: 1rem;
  }
  .metric {
    background: #fff;
    border-radius: 14px;
    padding: 1rem 1.1rem;
    border: 1px solid #e5e5ea;
  }
  .metric-label {
    font-size: 0.7rem;
    color: #999;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.4rem;
  }
  .metric-val {
    font-size: 1.6rem;
    font-weight: 600;
    color: #1a1a2e;
    line-height: 1;
  }
  .metric-denom { font-size: 0.9rem; color: #aaa; }
  .metric-sub { font-size: 0.72rem; margin-top: 0.3rem; }
  .up { color: #0F6E56; }
  .warn { color: #BA7517; }

  /* Layout rows */
  .row2 {
    display: grid;
    grid-template-columns: minmax(0,1.6fr) minmax(0,1fr);
    gap: 0.85rem;
    margin-bottom: 0.85rem;
  }
  .row3 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.85rem;
  }

  /* Card */
  .card {
    background: #fff;
    border-radius: 16px;
    border: 1px solid #e5e5ea;
    padding: 1rem;
  }
  .card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.9rem;
  }
  .card-title { font-size: 0.88rem; font-weight: 600; color: #1a1a2e; }
  .card-action { font-size: 0.75rem; color: #534AB7; cursor: default; }

  /* Mismatch list */
  .mismatch-list { display: flex; flex-direction: column; gap: 0.55rem; }
  .mismatch-item {
    display: flex;
    align-items: flex-start;
    gap: 0.65rem;
    padding: 0.65rem 0.75rem;
    background: #f7f7fa;
    border-radius: 10px;
  }
  .mismatch-icon { font-size: 1.2rem; flex-shrink: 0; margin-top: 0.1rem; }
  .mismatch-text { flex: 1; min-width: 0; }
  .mismatch-q { font-size: 0.8rem; font-weight: 600; color: #1a1a2e; }
  .mismatch-detail { font-size: 0.72rem; color: #777; margin-top: 0.15rem; line-height: 1.4; }
  .mismatch-time { font-size: 0.7rem; color: #bbb; flex-shrink: 0; margin-top: 0.1rem; }

  /* Tags */
  .tag {
    display: inline-flex;
    font-size: 0.65rem;
    padding: 0.15rem 0.45rem;
    border-radius: 999px;
    margin-left: 0.35rem;
    vertical-align: middle;
  }
  .tag-amber { background: #FAEEDA; color: #633806; }
  .tag-teal  { background: #E1F5EE; color: #0F6E56; }
  .tag-purple{ background: #EEEDFE; color: #3C3489; }

  /* Bar chart */
  .bar-group { display: flex; flex-direction: column; gap: 0.55rem; margin-bottom: 1rem; }
  .bar-row { display: flex; align-items: center; gap: 0.55rem; }
  .bar-label { font-size: 0.76rem; color: #555; width: 7.5rem; flex-shrink: 0; }
  .bar-track { flex: 1; height: 7px; background: #f0f0f5; border-radius: 999px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 999px; }
  .bar-pct { font-size: 0.76rem; font-weight: 600; color: #333; width: 2.5rem; text-align: right; flex-shrink: 0; }

  /* Trend chart */
  .trend-section { border-top: 1px solid #f0f0f5; padding-top: 0.75rem; }
  .section-divider {
    font-size: 0.68rem;
    font-weight: 600;
    color: #aaa;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.6rem;
  }
  .trend-bars { display: flex; align-items: flex-end; gap: 0.45rem; height: 72px; }
  .trend-bar-wrap { display: flex; flex-direction: column; align-items: center; gap: 0.3rem; flex: 1; }
  .trend-bar { width: 100%; border-radius: 3px 3px 0 0; }
  .trend-label { font-size: 0.65rem; color: #aaa; }

  /* Links card */
  .link-card {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.65rem 0.75rem;
    background: #f7f7fa;
    border-radius: 10px;
    margin-bottom: 0.55rem;
    gap: 0.5rem;
  }
  .link-meta { font-size: 0.68rem; color: #aaa; margin-bottom: 0.15rem; }
  .link-url { font-size: 0.78rem; color: #444; font-family: monospace; }
  .link-actions { display: flex; gap: 0.35rem; flex-shrink: 0; }
  .link-btn {
    font-size: 0.7rem;
    padding: 0.25rem 0.55rem;
    border: 1px solid #ddd;
    border-radius: 7px;
    background: #fff;
    color: #555;
    cursor: default;
  }
  .kiosk-note { font-size: 0.72rem; color: #aaa; margin-top: 0.4rem; }
  .generate-btn {
    font-size: 0.78rem;
    padding: 0.4rem 0.8rem;
    background: #534AB7;
    color: #EEEDFE;
    border: none;
    border-radius: 8px;
    cursor: default;
    font-weight: 500;
  }

  /* Questions card */
  .question-item {
    padding: 0.55rem 0;
    border-bottom: 1px solid #f0f0f5;
  }
  .question-item:last-child { border-bottom: none; }
  .q-count { font-size: 0.7rem; color: #aaa; margin-bottom: 0.2rem; }
  .q-text { font-size: 0.82rem; color: #1a1a2e; }

  /* Responsive */
  @media (max-width: 900px) {
    .metrics { grid-template-columns: repeat(2, 1fr); }
    .row2, .row3 { grid-template-columns: 1fr; }
  }
  @media (max-width: 480px) {
    .metrics { grid-template-columns: 1fr 1fr; }
    .main { padding: 0.75rem; }
    .topbar { padding: 0.75rem; }
  }
</style>
