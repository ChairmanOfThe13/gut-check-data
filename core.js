/* Line Lab core engine — shared by every front end.
   Data: data.json (stats, games with per-stat matchup z-scores, injuries).
   Profiles: {id,name,w:{statKey:0..10},inj:0..2,builtin?}
   Gut: {gameId:{m:homeMargin, ts, mode:'live'|'practice'}} + submitted weeks. */
(function (root) {
  const BE = 0.5238;               // break-even at -110
  const TEST_FROM = 2023;          // backtest window start (model trained 2017-2022)
  const L = {};

  // Where the hosted data lives (GitHub Pages). Empty = use the copy bundled with the app.
  L.DATA_URL = root.GC_DATA_URL || '';
  async function getJSON(u) { const r = await fetch(u, { cache: 'no-store' }); if (!r.ok) throw new Error(r.status); return r.json(); }
  L.load = async function (url) {
    let D = null, src = 'bundled';
    if (L.DATA_URL) { try { D = await getJSON(L.DATA_URL + 'data.json?t=' + Date.now()); src = 'online'; } catch (e) { console.warn('hosted data unavailable', e); } }
    if (!D) D = await getJSON(url || 'data.json');
    D.byId = {}; D.games.forEach((g, i) => { g.i = i; D.byId[g.id] = g; });
    D.keys = D.stats.map(s => s.key); D.source = src; D.lines_updated = null;
    {
      try {
        const LN = await getJSON((L.DATA_URL || '') + 'lines.json?t=' + Date.now());
        Object.entries(LN.lines || {}).forEach(([id, v]) => { const g = D.byId[id]; if (g && g.m == null && v.l != null) { g.l = v.l; g.lb = v.books; g.bk = v.bk || null; } });
        D.lines_updated = LN.updated;
      } catch (e) { console.warn('live lines unavailable', e); }
    }
    L.D = D; return D;
  };

  // weighted average of z-scores
  function score(D, p) {
    const w = D.keys.map(k => +(p.w[k] || 0)); const tot = w.reduce((a, b) => a + b, 0) || 1;
    return D.games.map(g => { let s = 0; for (let j = 0; j < w.length; j++) s += w[j] * g.z[j]; return s / tot / 100; });
  }
  function injd(g) { return (g.ia || 0) - (g.ih || 0); }

  // OLS: (y - inj) ~ a*(1-neutral) + b*s
  function fit(D, s, p, mask) {
    let sxx = 0, sxy = 0, syy = 0, s1 = 0, sy = 0, n = 0, s11 = 0, s1x = 0;
    const iw = +(p.inj ?? 1);
    D.games.forEach((g, i) => {
      if (!mask(g)) return;
      const y = g.m - iw * injd(g), h = 1 - g.n, x = s[i];
      s11 += h * h; s1x += h * x; sxx += x * x; s1 += h * y; sxy += x * y; n++;
    });
    const det = s11 * sxx - s1x * s1x || 1;
    return { a: (s1 * sxx - s1x * sxy) / det, b: (s11 * sxy - s1x * s1) / det, n };
  }

  // predicted home margin for every game. Backtest games use a fit on 2017-2022;
  // current-season games use a fit on every finished game before the current week.
  // stretch predictions to the betting line's range (mean/spread matched on the fit games)
  function scaler(D, raw, mask) {
    let n = 0, sp = 0, sl = 0; D.games.forEach((g, i) => { if (mask(g) && g.l != null) { n++; sp += raw[i]; sl += g.l; } });
    const mp = sp / n, ml = sl / n; let vp = 0, vl = 0;
    D.games.forEach((g, i) => { if (mask(g) && g.l != null) { vp += (raw[i] - mp) ** 2; vl += (g.l - ml) ** 2; } });
    const k = Math.sqrt(vl / (vp || 1)); return x => ml + (x - mp) * k;
  }
  L.predict = function (p) {
    const D = L.D, s = score(D, p), iw = +(p.inj ?? 1);
    const mT = g => g.m != null && g.s <= D.train_through, mL = g => g.m != null && (g.s < D.season || g.w < D.current_week);
    const fTest = fit(D, s, p, mT), fLive = fit(D, s, p, mL);
    const isLive = g => g.s === D.season && g.w >= D.current_week;
    let raw = D.games.map((g, i) => { const f = isLive(g) ? fLive : fTest; return f.a * (1 - g.n) + f.b * s[i] + iw * injd(g); });
    if (p.scale === 'market') { const sT = scaler(D, raw, mT), sL = scaler(D, raw, mL); raw = raw.map((x, i) => (isLive(D.games[i]) ? sL : sT)(x)); }
    return raw.map(x => Math.round(x * 10) / 10);
  };
  // which games a playbook is willing to bet
  L.allowed = function (p, g, edge) {
    if (!edge || g.l == null) return false;
    const dog = g.l === 0 ? false : (edge > 0) === (g.l < 0);   // betting the side getting points
    if (p.only === 'dog') return dog;
    if (p.only === 'roaddog') return dog && edge < 0;            // away team getting points
    return true;
  };

  // grade a list of {g, pred} against the line
  L.grade = function (rows, minEdge) {
    minEdge = minEdge || 0;
    const out = { w: 0, l: 0, p: 0, su: 0, suN: 0, ae: 0, aeN: 0, weeks: {}, seasons: {}, buckets: [0, 1.5, 3, 4.5].map(lo => ({ lo, w: 0, l: 0 })) };
    out.dog = 0;
    rows.forEach(({ g, pred, skip }) => {
      if (g.m == null) return;
      out.ae += Math.abs(pred - g.m); out.aeN++;
      if (pred !== 0 && g.m !== 0) { out.suN++; if (Math.sign(pred) === Math.sign(g.m)) out.su++; }
      if (g.l == null || skip) return;
      const edge = pred - g.l; if (Math.abs(edge) < minEdge || edge === 0) return;
      if (g.l !== 0 && (edge > 0) === (g.l < 0)) out.dog++;
      const c = g.m - g.l, wk = g.s + '-' + String(g.w).padStart(2, '0');
      const W = out.weeks[wk] || (out.weeks[wk] = { s: g.s, w: g.w, W: 0, L: 0, P: 0 });
      const S = out.seasons[g.s] || (out.seasons[g.s] = { W: 0, L: 0 });
      if (c === 0) { out.p++; W.P++; return; }
      const win = Math.sign(edge) === Math.sign(c);
      win ? (out.w++, W.W++, S.W++) : (out.l++, W.L++, S.L++);
      const b = out.buckets.filter(b => Math.abs(edge) >= b.lo).pop(); win ? b.w++ : b.l++;
    });
    const n = out.w + out.l;
    out.n = n; out.dogPct = (n + out.p) ? out.dog / (n + out.p) : null; out.pct = n ? out.w / n : null; out.units = Math.round((out.w - 1.1 * out.l) * 10) / 10;
    out.mae = out.aeN ? Math.round(out.ae / out.aeN * 100) / 100 : null;
    out.suPct = out.suN ? out.su / out.suN : null; out.be = BE;
    out.weekList = Object.values(out.weeks).sort((a, b) => a.s - b.s || a.w - b.w);
    return out;
  };

  // window: 'season' | 'all' (2023+)
  L.backtest = function (p, opt) {
    opt = opt || {}; const D = L.D, pr = L.predict(p);
    const from = opt.window === 'season' ? D.season : TEST_FROM;
    const rows = D.games.filter(g => g.s >= from && g.m != null).map(g => ({ g, pred: pr[g.i], skip: !L.allowed(p, g, Math.round((pr[g.i] - g.l) * 10) / 10) }));
    return L.grade(rows, opt.minEdge);
  };

  L.picks = function (p, week) {
    const D = L.D, pr = L.predict(p);
    return D.games.filter(g => g.s === D.season && g.w === week).map(g => { const x = L.pick(g, pr[g.i]); x.skip = !L.allowed(p, g, x.edge); return x; });
  };
  L.pick = function (g, pred) {
    const edge = g.l == null ? null : Math.round((pred - g.l) * 10) / 10;
    const side = edge == null ? null : edge > 0 ? g.h : g.a;
    let res = null;
    if (g.m != null && g.l != null && edge) { const c = g.m - g.l; res = c === 0 ? 'P' : Math.sign(c) === Math.sign(edge) ? 'W' : 'L'; }
    return { g, pred, edge, side, bet: side == null ? null : L.betText(g, side), res };
  };

  // team colors: [primary, secondary, text]
  L.TEAMC = {"ARI":["#97233f","#000000","#ffffff"],"ATL":["#a71930","#000000","#ffffff"],"BAL":["#241773","#000000","#ffffff"],"BUF":["#00338d","#c60c30","#ffffff"],"CAR":["#0085ca","#101820","#ffffff"],"CHI":["#0b162a","#c83803","#ffffff"],"CIN":["#000000","#fb4f14","#ffffff"],"CLE":["#311d00","#ff3c00","#ffffff"],"DAL":["#002244","#b0b7bc","#ffffff"],"DEN":["#002244","#fb4f14","#ffffff"],"DET":["#005a8b","#b0b7bc","#ffffff"],"GB":["#203731","#ffb612","#ffffff"],"HOU":["#03202f","#a71930","#ffffff"],"IND":["#002c5f","#a5acaf","#ffffff"],"JAX":["#000000","#006778","#ffffff"],"KC":["#e31837","#ffb612","#ffffff"],"LA":["#003594","#ffa300","#ffffff"],"LAC":["#0080c6","#ffc20e","#ffffff"],"LV":["#000000","#a5acaf","#ffffff"],"MIA":["#008e97","#fc4c02","#ffffff"],"MIN":["#4f2683","#ffc62f","#ffffff"],"NE":["#002244","#c60c30","#ffffff"],"NO":["#101820","#d3bc8d","#d3bc8d"],"NYG":["#0b2265","#a71930","#ffffff"],"NYJ":["#125740","#ffffff","#ffffff"],"PHI":["#004953","#a5acaf","#ffffff"],"PIT":["#000000","#ffb612","#ffffff"],"SEA":["#002244","#69be28","#ffffff"],"SF":["#aa0000","#b3995d","#ffffff"],"TB":["#d50a0a","#34302b","#ffffff"],"TEN":["#002244","#4b92db","#ffffff"],"WAS":["#5a1414","#ffb612","#ffffff"]};

  // kickoff label from schedule date + ET time ("20:20") -> "Wed, Nov 25 · 8:20 PM ET"
  L.kick = function (g) {
    const d = new Date(g.d + 'T12:00:00'), day = d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
    if (!g.t) return day; const [h, m] = g.t.split(':').map(Number);
    return `${day} · ${((h + 11) % 12) + 1}:${String(m).padStart(2, '0')} ${h < 12 ? 'AM' : 'PM'} ET`;
  };

  // ----- formatting (home margin convention: + means home wins by that much) -----
  L.sg = x => x > 0 ? '+' + x : x < 0 ? '−' + (-x) : 'PK';
  L.lineText = (g, m) => m == null ? '—' : m > 0 ? `${g.h} −${m}` : m < 0 ? `${g.a} −${-m}` : 'Pick’em';
  L.betText = (g, side) => side === g.h ? `${g.h} ${L.sg(-g.l)}` : `${g.a} ${L.sg(g.l)}`;
  L.pct = x => x == null ? '—' : (x * 100).toFixed(1) + '%';
  L.rec = r => `${r.w}–${r.l}${r.p ? '–' + r.p : ''}`;
  L.esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  L.weeks = () => [...new Set(L.D.games.filter(g => g.s === L.D.season).map(g => g.w))].sort((a, b) => a - b);
  L.weekGames = w => L.D.games.filter(g => g.s === L.D.season && g.w === w);

  // ----- Gut Instincts -----
  // gut = {picks:{gid:{m,mode,ts}}, submitted:{'2026-12':{mode,ts}}}
  L.gutGrade = function (gut, mode) {
    const rows = Object.entries(gut.picks || {}).filter(([, v]) => !mode || v.mode === mode)
      .map(([id, v]) => { const g0 = L.D.byId[id]; if (!g0 || !L.isRev(gut, g0, v)) return null;
        return { g: v.rl != null ? { ...g0, l: v.rl } : g0, pred: v.m }; }).filter(Boolean);
    return L.grade(rows);
  };
  // same-games comparison for a profile
  L.gradeOn = function (p, ids) {
    const pr = L.predict(p);
    return L.grade(ids.map(id => L.D.byId[id]).filter(Boolean).map(g => ({ g, pred: pr[g.i], skip: !L.allowed(p, g, Math.round((pr[g.i] - g.l) * 10) / 10) })));
  };
  L.reveal = function (gut, week) {
    return L.weekGames(week).filter(g => gut.picks[g.id] && g.l != null && L.isRev(gut, g, gut.picks[g.id])).map(g0 => {
      const v = gut.picks[g0.id], g = v.rl != null ? { ...g0, l: v.rl } : g0, m = v.m; const p = L.pick(g, m); p.diff = Math.round((m - g.l) * 10) / 10; return p;
    }).sort((a, b) => Math.abs(b.diff) - Math.abs(a.diff));
  };


  // ----- line shopping -----
  // Offshore books are not licensed in any US state; never shop them.
  const OFFSHORE = new Set(['betus', 'mybookieag', 'bovada', 'betonlineag', 'lowvig', 'betanysports', 'everygame', 'gtbets', 'intertops']);
  // side: home team abbreviation or away; books: array of keys to consider (null = all)
  L.bestLine = function (g, side, books) {
    if (!g.bk || !g.bk.length) return null;
    const home = side === g.h;
    const rows = g.bk.filter(b => !OFFSHORE.has(b.k) && (!books || !books.length || books.includes(b.k)))
      .map(b => ({ k: b.k, t: b.t, pt: home ? b.hp : b.ap, pr: home ? b.hpr : b.apr }))
      .filter(r => r.pt != null)
      .sort((a, b) => (b.pt - a.pt) || ((b.pr ?? -999) - (a.pr ?? -999)));   // more points first, then better price
    if (!rows.length) return null;
    return { ...rows[0], next: rows[1] || null, n: rows.length };
  };
  L.price = p => p == null ? '' : (p > 0 ? '+' + p : String(p).replace('-', '−'));
  L.books = function () {   // every sportsbook seen in the current lines
    const m = {}; (L.D.games || []).forEach(g => (g.bk || []).forEach(b => { if (!OFFSHORE.has(b.k)) m[b.k] = b.t; })); return Object.entries(m).sort((a, b) => a[1].localeCompare(b[1]));
  };
  // what line shopping was worth on graded gut picks
  L.shopStats = function (gut) {
    const out = { n: 0, better: 0, pts: 0, saved: 0 };
    Object.entries(gut.picks || {}).forEach(([id, v]) => {
      const best = v.bestB || (v.dec ? null : v.best), g = L.D.byId[id]; if (!g || !best || v.rl == null || !L.isRev(gut, g, v)) return;
      const home = best.side === g.h, cons = home ? -v.rl : v.rl, gain = best.pt - cons;
      out.n++; if (gain > 0) { out.better++; out.pts += gain; }
      if (g.m != null && gain > 0) {   // did the extra points turn a loss into a push/win, or a push into a win?
        const marg = home ? g.m : -g.m, a = marg + cons, b = marg + best.pt;
        if ((a < 0 && b >= 0) || (a === 0 && b > 0)) out.saved++;
      }
    });
    return out;
  };


  // ----- per-game state (older whole-week reveals/locks still count) -----
  L.wk = g => g.s + '-' + g.w;
  L.isRev = (gut, g, v) => !!(v && g && (v.rev || (gut.submitted || {})[L.wk(g)]));
  L.isLk = (gut, g, v) => !!(v && g && (v.lk || (gut.locked || {})[L.wk(g)]));
  L.started = g => {   // kickoff has passed (schedule times are Eastern)
    if (!g.t) return false; const mo = +g.d.slice(5, 7), off = (mo >= 11 || mo <= 3) ? '-05:00' : '-04:00';
    return Date.now() >= new Date(`${g.d}T${g.t}:00${off}`).getTime();
  };

  // ----- bets made after the reveal: dec = 'bet' (your read) | 'fade' (opposite) | 'pass'; u = units -----
  L.readSide = (g, m, l) => { const e = m - l; return e > 0 ? g.h : e < 0 ? g.a : null; };
  L.other = (g, side) => side === g.h ? g.a : g.h;
  L.betSide = function (g, v) {
    if (v.dec === 'home') return g.h; if (v.dec === 'away') return g.a;   // you matched Vegas exactly and picked a side
    const l = v.rl ?? g.l, read = L.readSide(g, v.m, l); if (!read || !v.dec || v.dec === 'pass') return null;
    return v.dec === 'fade' ? L.other(g, read) : read;
  };
  L.cover = function (g, side, l) {   // 'W' | 'L' | 'P' | null for a side at line l (home-favored positive)
    if (g.m == null || !side) return null; const c = (side === g.h ? 1 : -1) * (g.m - l); return c > 0 ? 'W' : c < 0 ? 'L' : 'P';
  };
  L.betRecord = function (gut, opt) {
    opt = opt || {}; const r = { w: 0, l: 0, p: 0, units: 0, risked: 0, n: 0, open: 0 };
    Object.entries(gut.picks || {}).forEach(([id, v]) => {
      const g = L.D.byId[id]; if (!g || !L.isLk(gut, g, v)) return;
      if (opt.mode && v.mode !== opt.mode) return; if (opt.week && g.w !== opt.week) return;
      if (opt.dec === 'match' ? !(v.dec === 'home' || v.dec === 'away') : (opt.dec && v.dec !== opt.dec)) return;
      const l = v.rl ?? g.l, side = opt.dec === 'pass' ? L.readSide(g, v.m, l) : L.betSide(g, v);
      if (!side) return; const u = opt.dec === 'pass' ? 1 : (v.u || 1), res = L.cover(g, side, l);
      if (!res) { r.open++; return; }
      r.n++; r.risked += u; if (res === 'W') { r.w++; r.units += u; } else if (res === 'L') { r.l++; r.units -= 1.1 * u; } else r.p++;
    });
    r.units = Math.round(r.units * 10) / 10; r.pct = (r.w + r.l) ? r.w / (r.w + r.l) : null; r.be = 0.5238; return r;
  };
  // your raw reads split by how far your line was from the market's
  L.gapBuckets = function (gut, mode) {
    const B = [[0, 3, '0–3 pts'], [3, 7, '3–7 pts'], [7, 99, '7+ pts']].map(([lo, hi, t]) => ({ lo, hi, t, w: 0, l: 0 }));
    Object.entries(gut.picks || {}).forEach(([id, v]) => {
      const g = L.D.byId[id]; if (!g || g.m == null || !L.isRev(gut, g, v) || (mode && v.mode !== mode)) return;
      const l = v.rl ?? g.l; if (l == null) return; const gap = Math.abs(v.m - l), side = L.readSide(g, v.m, l), res = L.cover(g, side, l);
      if (!res || res === 'P') return; const b = B.find(b => gap >= b.lo && gap < b.hi) || B[2]; res === 'W' ? b.w++ : b.l++;
    });
    return B;
  };

  // ----- storage: Claude artifact db (per-person, private) with localStorage fallback -----
  const LS = 'linelab.v1';
  L.store = {
    db: null, uid: null, state: { profiles: [], gut: { picks: {}, submitted: {} }, books: null },
    async init(onChange) {
      try { const s = JSON.parse(localStorage.getItem(LS) || 'null'); if (s) this.state = s; } catch (e) { }
      this.state.gut = this.state.gut || { picks: {}, submitted: {} };
      onChange && onChange();
      try {
        if (!(root.claude && root.claude.use)) return;
        const db = await root.claude.use('db'), user = await root.claude.use('user');
        if (!db || !user) return;
        this.uid = await user.id(); this.db = db;
        this.ref = db.collection('data/users/' + this.uid).doc('linelab');
        const snap = await Promise.race([this.ref.get(), new Promise((_, rej) => setTimeout(() => rej(new Error('timeout')), 10000))]);
        const v = snap && (snap.data ? snap.data() : snap);
        if (v && v.profiles) {
          const remote = JSON.parse(JSON.stringify(v)), local = this.state;
          if (this._dirty) {  // user acted before the account copy arrived: merge, local wins
            const byId = {}; (remote.profiles || []).forEach(p => byId[p.id] = p); (local.profiles || []).forEach(p => byId[p.id] = p);
            remote.profiles = Object.values(byId);
            remote.gut = { picks: { ...(remote.gut && remote.gut.picks), ...local.gut.picks }, submitted: { ...(remote.gut && remote.gut.submitted), ...local.gut.submitted } };
          }
          this.state = remote; this.state.gut = this.state.gut || { picks: {}, submitted: {} }; this.state.gut.picks = this.state.gut.picks || {}; this.state.gut.submitted = this.state.gut.submitted || {}; this.state.profiles = this.state.profiles || []; this._loaded = true; this._ls(); if (this._dirty) this.save(); onChange && onChange(); }
        else { this._loaded = true; if (this.state.profiles.length || Object.keys(this.state.gut.picks).length) this.save(); }
      } catch (e) { this.db = null; console.warn('db unavailable, using this device only', e); }
    },
    _ls() { try { localStorage.setItem(LS, JSON.stringify(this.state)); } catch (e) { } },
    _busy: false, _again: false,
    async save() {
      this._ls();
      if (!this.db || !this._loaded) { this._dirty = true; return; }
      if (this._busy) { this._again = true; return; }
      this._busy = true;
      try { await this.ref.set(JSON.parse(JSON.stringify(this.state))); } catch (e) { console.warn(e); }
      this._busy = false; if (this._again) { this._again = false; this.save(); }
    },
    synced() { return !!this.db; },
    profiles() { return [...L.D.presets, ...this.state.profiles]; },
    upsertProfile(p) {
      if (!p.id || p.builtin) p = { ...p, id: 'u' + Date.now().toString(36), builtin: false };
      const i = this.state.profiles.findIndex(x => x.id === p.id);
      const arr = [...this.state.profiles]; i >= 0 ? arr[i] = p : arr.push(p); this.state.profiles = arr;
      this.save(); return p;
    },
    deleteProfile(id) { this.state.profiles = this.state.profiles.filter(p => p.id !== id); this.save(); },
    deleteAllProfiles() { this.state.profiles = []; this.save(); },
    setGut(gid, m, mode) { const g = this.state.gut; g.picks = { ...g.picks, [gid]: { m, mode, ts: Date.now() } }; this.save(); },
    submitWeek(week, mode) {
      const g = this.state.gut, p = { ...g.picks };
      L.weekGames(week).forEach(x => {
        if (!p[x.id] || x.l == null) return;
        const e = p[x.id].m - x.l, side = e > 0 ? x.h : e < 0 ? x.a : null;
        const b = side ? L.bestLine(x, side, this.state.books) : null;
        p[x.id] = { ...p[x.id], rl: x.l, best: b ? { side, k: b.k, t: b.t, pt: b.pt, pr: b.pr } : null };   // line you saw at reveal
      });
      g.picks = p; g.submitted = { ...g.submitted, [L.D.season + '-' + week]: { mode, ts: Date.now() } }; this.save();
    },
    resetWeek(week) {
      const g = this.state.gut, p = { ...g.picks }, s = { ...g.submitted };
      L.weekGames(week).forEach(x => delete p[x.id]); delete s[L.D.season + '-' + week];
      const lk = { ...(g.locked || {}) }; delete lk[L.D.season + '-' + week];
      g.picks = p; g.submitted = s; g.locked = lk; this.save();
    },
    resetGut() { this.state.gut = { picks: {}, submitted: {}, locked: {} }; this.save(); },
    setDecision(gid, dec, u) { const g = this.state.gut, v = g.picks[gid]; if (!v) return; g.picks = { ...g.picks, [gid]: { ...v, dec, u: dec === 'pass' ? undefined : (u || v.u || 1) } }; this.save(); },
    lockBets(week) {
      const g = this.state.gut, p = { ...g.picks };
      L.weekGames(week).forEach(x => {
        const v = p[x.id]; if (!v || v.rl == null) return;
        const dec = v.dec || 'pass', nv = { ...v, dec };
        const side = L.betSide(x, nv), b = side ? L.bestLine(x, side, this.state.books) : null;
        nv.bestB = b ? { side, k: b.k, t: b.t, pt: b.pt, pr: b.pr } : null; p[x.id] = nv;
      });
      g.picks = p; g.locked = { ...(g.locked || {}), [L.D.season + '-' + week]: Date.now() }; this.save();
    },
    revealGame(gid) {
      const g = L.D.byId[gid], gs = this.state.gut, v = gs.picks[gid]; if (!g || !v || g.l == null || v.rev) return;
      const e = v.m - g.l, side = e > 0 ? g.h : e < 0 ? g.a : null, b = side ? L.bestLine(g, side, this.state.books) : null;
      gs.picks = { ...gs.picks, [gid]: { ...v, rl: g.l, rev: Date.now(), best: b ? { side, k: b.k, t: b.t, pt: b.pt, pr: b.pr } : null } }; this.save();
    },
    lockGame(gid) {
      const g = L.D.byId[gid], gs = this.state.gut, v = gs.picks[gid]; if (!g || !v || !L.isRev(gs, g, v) || v.lk) return;
      const nv = { ...v, dec: v.dec || 'pass', lk: Date.now() }; if (nv.rl == null) nv.rl = g.l;
      const side = L.betSide(g, nv), b = side ? L.bestLine(g, side, this.state.books) : null;
      nv.bestB = b ? { side, k: b.k, t: b.t, pt: b.pt, pr: b.pr } : null;
      gs.picks = { ...gs.picks, [gid]: nv }; this.save();
    },
    isLocked(week) { return !!(this.state.gut.locked && this.state.gut.locked[L.D.season + '-' + week]); },
    setBooks(arr) { this.state.books = arr && arr.length ? [...arr] : null; this.save(); },
    isSubmitted(week) { return !!this.state.gut.submitted[L.D.season + '-' + week]; },
    gut() { return this.state.gut; }
  };

  // leaderboard: every profile + Gut Instincts
  L.board = function (opt) {
    const rows = L.store.profiles().map(p => ({ p, r: L.backtest(p, opt) }));
    const gut = L.store.gut();
    const gl = L.gutGrade(gut, 'live'), gp = L.gutGrade(gut, 'practice');
    return { rows: rows.sort((a, b) => (b.r.pct ?? 0) - (a.r.pct ?? 0)), gutLive: gl, gutPractice: gp };
  };

  root.LineLab = L;
})(window);
