/* F1 & LoL Quant — renders data/*.json written by the Python pipeline.
   Plain JS, no build step. All names coming from data are inserted with
   textContent (via h()), never innerHTML. */
(function () {
  'use strict';

  var state = { tz: null, data: {}, recordSport: 'lol', lolLeagues: null };

  // ------------------------------------------------------------ helpers
  function $(s) { return document.querySelector(s); }

  function h(tag, props) {
    var el = document.createElement(tag);
    if (props) {
      for (var k in props) {
        var v = props[k];
        if (v == null || v === false) continue;
        if (k === 'class') el.className = v;
        else if (k === 'text') el.textContent = v;
        else if (k === 'style') el.setAttribute('style', v);
        else if (k.slice(0, 2) === 'on') el.addEventListener(k.slice(2), v);
        else el.setAttribute(k, v === true ? '' : v);
      }
    }
    for (var i = 2; i < arguments.length; i++) append(el, arguments[i]);
    return el;
  }

  function append(el, c) {
    if (c == null || c === false) return;
    if (Array.isArray(c)) { c.forEach(function (x) { append(el, x); }); return; }
    el.appendChild(typeof c === 'object' ? c : document.createTextNode(String(c)));
  }

  function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }

  var SVGNS = 'http://www.w3.org/2000/svg';
  function s(tag, attrs) {
    var el = document.createElementNS(SVGNS, tag);
    for (var k in attrs || {}) el.setAttribute(k, attrs[k]);
    return el;
  }

  function isNum(x) { return typeof x === 'number' && isFinite(x); }
  function pct(p, d) { return isNum(p) ? (p * 100).toFixed(d == null ? 0 : d) + '%' : '–'; }
  // Probabilities: never print a rounded 100% or 0% for something uncertain.
  function prob(p, d) {
    if (!isNum(p)) return '–';
    d = d == null ? 0 : d;
    var lim = 0.5 * Math.pow(10, -d) / 100;
    if (p >= 1 - lim && p < 1) return '>' + (100 - Math.pow(10, -d)).toFixed(d) + '%';
    if (p > 0 && p < lim) return '<' + Math.pow(10, -d).toFixed(d) + '%';
    return pct(p, d);
  }
  function cents(p) { return isNum(p) ? Math.round(p * 100) + '¢' : '–'; }
  function num(x, d) { return isNum(x) ? x.toFixed(d || 0) : '–'; }
  function signed(x, d) { return isNum(x) ? (x > 0 ? '+' : x < 0 ? '−' : '±') + Math.abs(x).toFixed(d == null ? 3 : d) : '–'; }

  function edgeEl(p, ask) {
    if (!isNum(p) || !isNum(ask) || ask <= 0 || ask >= 1) return null;
    var e = Math.round((p - ask) * 100);
    return h('span', { class: 'edge ' + (e > 0 ? 'pos' : 'neg'),
      title: 'Model probability minus the price to buy YES (before fees)' },
      (e > 0 ? '+' : e < 0 ? '−' : '±') + Math.abs(e) + '¢');
  }

  // ------------------------------------------------------------- time
  function fmt(iso, opts) {
    if (!iso) return '–';
    var d = new Date(iso);
    if (isNaN(d)) return '–';
    var o = Object.assign({ timeZone: state.tz }, opts);
    try { return new Intl.DateTimeFormat(undefined, o).format(d); }
    catch (e) { return d.toLocaleString(); }
  }
  function fmtDay(iso) { return fmt(iso, { weekday: 'long', month: 'short', day: 'numeric' }); }
  function fmtTime(iso) { return fmt(iso, { hour: '2-digit', minute: '2-digit' }); }
  function fmtDate(iso) { return fmt(iso, { month: 'short', day: 'numeric' }); }
  function fmtFull(iso) { return fmt(iso, { weekday: 'short', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }); }

  function until(iso) {
    var ms = new Date(iso) - Date.now();
    if (!isFinite(ms)) return '–';
    if (ms <= 0) return 'started';
    var m = Math.floor(ms / 60000), d = Math.floor(m / 1440), hr = Math.floor((m % 1440) / 60);
    return d > 0 ? d + 'd ' + hr + 'h' : hr > 0 ? hr + 'h ' + (m % 60) + 'm' : (m % 60) + 'm';
  }
  function ago(iso) {
    var m = Math.round((Date.now() - new Date(iso)) / 60000);
    if (!isFinite(m)) return '–';
    return m < 60 ? m + ' min ago' : m < 1440 ? Math.round(m / 60) + ' h ago' : Math.round(m / 1440) + ' d ago';
  }

  // --------------------------------------------------------- tooltip
  var tip = $('#tooltip');
  function showTip(evt, title, rows) {
    clear(tip);
    if (title) tip.appendChild(h('div', { class: 'title' }, title));
    rows.forEach(function (r) {
      tip.appendChild(h('div', { class: 'row' },
        r.color ? h('span', { class: 'key-line', style: 'background:' + r.color }) : null,
        h('strong', null, r.value), h('span', { class: 'muted' }, r.label || '')));
    });
    tip.hidden = false;
    var x = evt.clientX + 14, y = evt.clientY + 14;
    var w = tip.offsetWidth, hh = tip.offsetHeight;
    if (x + w > window.innerWidth - 8) x = evt.clientX - w - 14;
    if (y + hh > window.innerHeight - 8) y = evt.clientY - hh - 14;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  }
  function hideTip() { tip.hidden = true; }

  // ------------------------------------------------------------- bars
  function probBar(p, mid) {
    var bar = h('span', { class: 'bar', role: 'img',
      'aria-label': 'Model ' + pct(p) + (isNum(mid) ? ', market ' + cents(mid) : '') });
    bar.appendChild(h('i', { style: 'width:' + (isNum(p) ? Math.max(p * 100, 1) : 0) + '%' }));
    if (isNum(mid)) bar.appendChild(h('b', { style: 'left:calc(' + (mid * 100) + '% - 1px)' }));
    return bar;
  }

  function table(el, headers, rows, caption) {
    clear(el);
    var thead = h('thead', null, h('tr', null, headers.map(function (c) {
      return h('th', { class: c.r ? 'r' : null, scope: 'col' }, c.t);
    })));
    var tbody = h('tbody', null, rows);
    el.appendChild(thead); el.appendChild(tbody);
    if (caption) el.appendChild(h('caption', null, caption));
    if (!rows.length) tbody.appendChild(h('tr', null, h('td', { colspan: headers.length, class: 'muted' }, 'Nothing yet.')));
  }

  function legend() {
    return h('p', { class: 'note' },
      h('span', { class: 'legend-inline' }, h('span', { class: 'key-bar' }), 'Model probability'),
      h('span', { class: 'legend-inline' }, h('span', { class: 'key-tick' }), 'Kalshi price (mid)'),
      h('span', { class: 'legend-inline' }, 'Edge = model − ask, before fees'));
  }

  // ------------------------------------------------------- the decision
  // The model's call: who it picks and, where a Kalshi price is on the page,
  // whether buying is worth it after fees (net edge >= meta.edge_min).
  function kalshiFee(price) {
    var rate = (state.data.meta || {}).fee_rate;
    return Math.ceil((rate == null ? 0.07 : rate) * price * (1 - price) * 100) / 100;
  }
  function netEdge(p, ask) {
    return isNum(p) && isNum(ask) && ask > 0 && ask < 1 ? p - ask - kalshiFee(ask) : null;
  }
  function edgeMin() { return (state.data.meta || {}).edge_min || 0.05; }

  function lolPick(m, k) {
    if (!isNum(m.p1)) return null;
    var one = m.p1 >= 0.5, name = one ? m.team1 : m.team2, p = one ? m.p1 : 1 - m.p1;
    var conf = p >= 0.75 ? 'strong' : p >= 0.6 ? 'clear' : 'lean';
    var parts = [h('span', { class: 'pick-label' }, 'Model pick'), h('strong', null, name + ' ' + prob(p)),
      h('span', { class: 'pick-conf ' + conf },
        conf === 'strong' ? 'strong favourite' : conf === 'clear' ? 'clear favourite' : 'lean, close match')];
    if (k) {
      var e1 = netEdge(m.p1, k.t1 && k.t1.yes_ask), e2 = netEdge(1 - m.p1, k.t2 && k.t2.yes_ask);
      var best = e1 != null && (e2 == null || e1 >= e2) ? [m.team1, e1, k.t1.yes_ask]
        : e2 != null ? [m.team2, e2, k.t2.yes_ask] : null;
      parts.push(best && best[1] >= edgeMin()
        ? h('span', { class: 'pick-value' }, 'Value: ' + best[0] + ' at ' + cents(best[2]) + ' (' + signed(best[1] * 100, 0) + '¢ after fees)')
        : h('span', { class: 'muted' }, 'no value at current Kalshi prices'));
    }
    return h('div', { class: 'pick' }, parts);
  }

  function f1Pick(n) {
    var ds = (n.drivers || []).filter(function (x) { return isNum(x.p_win); });   // sorted by p_win
    if (!ds.length) return null;
    var podium = ds.slice().sort(function (a, b) { return (b.p_podium || 0) - (a.p_podium || 0); }).slice(0, 3);
    var priced = ds.filter(function (x) { return x.kalshi; });
    var value = priced.map(function (x) { return { x: x, e: netEdge(x.p_win, x.kalshi.yes_ask), ask: x.kalshi.yes_ask }; })
      .filter(function (v) { return v.e != null && v.e >= edgeMin(); })
      .sort(function (a, b) { return b.e - a.e; });
    return h('div', { class: 'pick' },
      h('span', { class: 'pick-label' }, 'Model pick to win'), h('strong', null, ds[0].code + ' ' + prob(ds[0].p_win)),
      h('span', { class: 'muted' }, 'podium: ' + podium.map(function (x) { return x.code + ' ' + prob(x.p_podium); }).join(', ')),
      value.length
        ? h('span', { class: 'pick-value' }, 'Value: ' + value.slice(0, 3).map(function (v) {
            return v.x.code + ' at ' + cents(v.ask) + ' (' + signed(v.e * 100, 0) + '¢)'; }).join(', '))
        : priced.length ? h('span', { class: 'muted' }, 'no value at current Kalshi prices') : null);
  }

  function pickNote(text) {
    return h('p', { class: 'note' }, text + ' "Value" means model probability minus the Kalshi ask minus the fee is at least ' +
      cents(edgeMin()) + '. These are forecasts, not betting advice.');
  }

  // ------------------------------------------------------------- LoL
  function renderLoL() {
    var d = state.data.lol;
    var box = clear($('#lol-upcoming'));
    if (!d) { box.appendChild(h('div', { class: 'empty' }, 'No LoL data yet.')); return; }

    var leagues = [];
    (d.upcoming || []).forEach(function (m) { if (leagues.indexOf(m.league) < 0) leagues.push(m.league); });
    var fs = $('#lol-leagues');
    if (!state.lolLeagues) {
      state.lolLeagues = {};
      Array.prototype.slice.call(fs.querySelectorAll('label')).forEach(function (l) { l.remove(); });
      leagues.forEach(function (lg) {
        state.lolLeagues[lg] = true;
        var cb = h('input', { type: 'checkbox', checked: true, onchange: function () {
          state.lolLeagues[lg] = cb.checked; applyLoLFilters(); } });
        fs.appendChild(h('label', { class: 'chip' }, cb, lg));
      });
    }

    box.appendChild(legend());
    var live = d.backtest && d.backtest.metrics && d.model && d.backtest.metrics[d.model.chosen];
    if (live && live.all && isNum(live.all.accuracy)) {
      box.appendChild(pickNote('Model pick = the team the model rates likelier. In the walk-forward backtest the pick won ' +
        pct(live.all.accuracy) + ' of ' + live.all.n + ' matches.'));
    }
    var byDay = {}, order = [];
    (d.upcoming || []).forEach(function (m) {
      var k = fmtDay(m.start);
      if (!byDay[k]) { byDay[k] = []; order.push(k); }
      byDay[k].push(m);
    });
    if (!order.length) box.appendChild(h('div', { class: 'empty' }, 'No matches scheduled in the next two weeks.'));
    order.forEach(function (day) {
      var list = h('div', { class: 'match-list' });
      box.appendChild(h('div', { class: 'day-group' }, h('div', { class: 'day-label' }, day), list));
      byDay[day].forEach(function (m) { list.appendChild(matchCard(m)); });
    });

    var rows = (d.results || []).map(function (r) {
      var fav = isNum(r.p1_pre) ? (r.p1_pre >= 0.5 ? 1 : 2) : null;
      var favName = fav === 1 ? r.team1 : fav === 2 ? r.team2 : '–';
      var favP = fav === 1 ? r.p1_pre : fav === 2 ? 1 - r.p1_pre : null;
      var ok = fav != null && fav === r.winner;
      return h('tr', null,
        h('td', { class: 'num' }, fmtDate(r.start)), h('td', null, r.league),
        h('td', null, h('span', { style: r.winner === 1 ? 'font-weight:650' : '' }, r.team1), ' vs ',
          h('span', { style: r.winner === 2 ? 'font-weight:650' : '' }, r.team2)),
        h('td', { class: 'r' }, (r.score1 != null ? r.score1 + '–' + r.score2 : '–') + ' (Bo' + r.best_of + ')'),
        h('td', null, favName + ' ', h('span', { class: 'muted' }, prob(favP))),
        h('td', null, fav == null ? '–' : h('span', { class: ok ? 'hit' : 'miss' }, ok ? '✓ hit' : '✗ miss')));
    });
    table($('#lol-results'), [{ t: 'Date' }, { t: 'League' }, { t: 'Match' }, { t: 'Score', r: 1 }, { t: 'Pre-match favourite' }, { t: '' }], rows);

    var rk = (d.rankings || []).slice(0, 40).map(function (t, i) {
      return h('tr', null, h('td', { class: 'r' }, i + 1), h('td', { style: 'font-weight:600' }, t.team),
        h('td', { class: 'muted' }, (t.league || '').toUpperCase()), h('td', { class: 'r' }, t.elo),
        h('td', { class: 'r' }, t.change_30d == null ? '–' : h('span', { class: t.change_30d > 0 ? 'edge pos' : 'muted' }, signed(t.change_30d, 0))),
        h('td', { class: 'r muted' }, t.games));
    });
    var unranked = d.coverage && d.coverage.tier2_unranked;
    table($('#lol-rankings'), [{ t: '#', r: 1 }, { t: 'Team' }, { t: 'League' }, { t: 'Elo', r: 1 }, { t: '30 days', r: 1 }, { t: 'Games', r: 1 }], rk,
      unranked ? unranked + ' active teams from second-tier leagues (national and academy leagues) are not ranked: ' +
        'they almost never play the major regions, so their Elo is on a different scale.' : null);
    applyLoLFilters();
  }

  function matchCard(m) {
    var k = m.kalshi;
    var started = new Date(m.start) <= Date.now();
    var card = h('article', { class: 'match' + (m.intl ? ' intl' : ''),
      'data-league': m.league, 'data-kalshi': k ? '1' : '0',
      'data-text': (m.team1 + ' ' + m.team2 + ' ' + m.league + ' ' + (m.code1 || '') + ' ' + (m.code2 || '')).toLowerCase() });
    card.appendChild(h('div', { class: 'match-when' },
      h('strong', null, fmtTime(m.start)),
      m.state === 'inProgress' ? h('span', { class: 'live' }, '● LIVE') : started ? 'starting' : 'in ' + until(m.start)));
    var body = h('div', null, h('div', { class: 'badges' },
      h('span', { class: 'badge' + (m.intl ? ' solid' : '') }, m.league),
      m.block ? h('span', { class: 'badge' }, m.block) : null,
      h('span', { class: 'badge' }, 'Bo' + m.best_of),
      k ? h('span', { class: 'badge', title: k.event }, 'Kalshi') : null));
    if (m.tbd) {
      // Teams not decided yet: a schedule slot, no forecast until lolesports fills it in.
      card.className += ' tbd';
      body.appendChild(h('div', { class: 'tbd-line' }, h('span', { class: 'team-name' }, m.team1 + ' vs ' + m.team2),
        h('span', { class: 'muted' }, 'Teams set by earlier results · forecast once known')));
      card.appendChild(body);
      return card;
    }
    [[m.team1, m.code1, m.elo1, m.p1, k && k.t1], [m.team2, m.code2, m.elo2, isNum(m.p1) ? 1 - m.p1 : null, k && k.t2]]
      .forEach(function (x) {
        var q = x[4];
        body.appendChild(h('div', { class: 'side' },
          h('div', { class: 'team-name', title: x[0] }, x[0], h('small', null, 'Elo ' + num(x[2]))),
          h('div', { class: 'prob' }, probBar(x[3], q && q.mid), h('span', { class: 'val' }, prob(x[3]))),
          h('div', { class: 'mkt' }, q ? ['Kalshi ', cents(q.mid), h('span', { class: 'muted' }, ' · ask ' + cents(q.yes_ask)), edgeEl(x[3], q.yes_ask)]
                                       : h('span', { class: 'muted' }, 'no market'))));
      });
    body.appendChild(lolPick(m, k));
    card.appendChild(body);
    return card;
  }

  function applyLoLFilters() {
    var q = ($('#lol-search').value || '').trim().toLowerCase();
    var only = $('#lol-kalshi-only').checked;
    document.querySelectorAll('#lol-upcoming .match').forEach(function (c) {
      var show = (!state.lolLeagues || state.lolLeagues[c.getAttribute('data-league')] !== false) &&
        (!only || c.getAttribute('data-kalshi') === '1') &&
        (!q || c.getAttribute('data-text').indexOf(q) >= 0);
      c.hidden = !show;
    });
    document.querySelectorAll('#lol-upcoming .day-group').forEach(function (g) {
      g.hidden = !g.querySelector('.match:not([hidden])');
    });
  }

  // -------------------------------------------------------------- F1
  function renderF1() {
    var d = state.data.f1;
    var nextBox = clear($('#f1-next'));
    if (!d) { nextBox.appendChild(h('div', { class: 'empty' }, 'No F1 data yet.')); return; }
    var n = d.next_race;
    if (n) {
      nextBox.appendChild(h('div', { class: 'race-card' },
        h('div', { style: 'flex:1 1 260px' },
          h('div', { class: 'badges' }, h('span', { class: 'badge solid' }, 'Round ' + n.round),
            h('span', { class: 'badge' }, n.stage === 'post_quali' ? 'After qualifying' : 'Before qualifying'),
            n.sprint_time ? h('span', { class: 'badge' }, 'Sprint weekend') : null,
            h('span', { class: 'badge' }, 'model: ' + n.model)),
          h('h2', null, n.name), h('p', { class: 'sub' }, n.circuit + ' · ' + n.country), f1Pick(n)),
        n.quali_time ? h('div', { class: 'countdown' }, 'Qualifying', h('strong', { 'data-until': n.quali_time }, until(n.quali_time)), fmtFull(n.quali_time)) : null,
        h('div', { class: 'countdown' }, 'Race', h('strong', { 'data-until': n.race_time }, until(n.race_time)), fmtFull(n.race_time))));
      nextBox.appendChild(legend());
      var fb = d.backtest || {}, st = fb.metrics && fb.chosen && fb.metrics[n.stage] && fb.metrics[n.stage][fb.chosen[n.stage]];
      if (st && isNum(st.favourite_won)) {
        nextBox.appendChild(pickNote('In the backtest the model\'s favourite won ' + pct(st.favourite_won) + ' of ' + st.races +
          ' races at this stage (' + (n.stage === 'post_quali' ? 'after qualifying' : 'before qualifying') + ').'));
      }
      var rows = n.drivers.map(function (x, i) {
        var q = x.kalshi;
        return h('tr', null, h('td', { class: 'r muted' }, i + 1),
          h('td', null, h('strong', null, x.code), ' ', h('span', { class: 'muted' }, x.name)),
          h('td', null, x.team), h('td', { class: 'r' }, x.quali == null ? '–' : 'P' + x.quali),
          h('td', null, probBar(x.p_win, q && q.mid), h('strong', { class: 'num' }, prob(x.p_win, x.p_win < 0.1 ? 1 : 0))),
          h('td', { class: 'r' }, prob(x.p_podium)), h('td', { class: 'r' }, prob(x.p_points)),
          h('td', { class: 'r' }, num(x.exp_pos, 1)),
          h('td', { class: 'r' }, q ? cents(q.yes_bid) + ' / ' + cents(q.yes_ask) : h('span', { class: 'muted' }, '–')),
          h('td', { class: 'r' }, q ? edgeEl(x.p_win, q.yes_ask) : null));
      });
      table($('#f1-drivers'), [{ t: '#', r: 1 }, { t: 'Driver' }, { t: 'Team' }, { t: 'Quali', r: 1 }, { t: 'Win' },
        { t: 'Podium', r: 1 }, { t: 'Points', r: 1 }, { t: 'Avg pos', r: 1 }, { t: 'Kalshi bid/ask', r: 1 }, { t: 'Edge', r: 1 }], rows,
        'Podium and points chances come from 20,000 simulated races (Plackett–Luce). Kalshi lists only race-winner markets.');
    } else {
      nextBox.appendChild(h('div', { class: 'empty' }, 'Season complete — no upcoming race.'));
      clear($('#f1-drivers'));
    }

    var lr = d.last_race;
    $('#f1-last-title').textContent = lr ? 'Last race · ' + lr.name : 'Last race';
    table($('#f1-last'), [{ t: 'Pos', r: 1 }, { t: 'Driver' }, { t: 'Team' }, { t: 'Grid', r: 1 }, { t: 'Status' }, { t: 'Pts', r: 1 }, { t: 'Model win % before', r: 1 }],
      (lr ? lr.results : []).map(function (x) {
        return h('tr', { class: x.pos === 1 ? 'win' : null }, h('td', { class: 'r' }, x.pos),
          h('td', null, h('strong', null, x.code), ' ', h('span', { class: 'muted' }, x.name)), h('td', null, x.team),
          h('td', { class: 'r' }, x.grid || 'PL'), h('td', { class: 'muted' }, x.status), h('td', { class: 'r' }, num(x.points)),
          h('td', { class: 'r' }, prob(x.p_win_pre, 1)));
      }), 'Pre-race win probabilities are walk-forward backtest replays (post-qualifying model fitted only on earlier races).');

    var st = d.standings || {};
    table($('#f1-std-drivers'), [{ t: '#', r: 1 }, { t: 'Driver' }, { t: 'Team' }, { t: 'Wins', r: 1 }, { t: 'Pts', r: 1 }],
      (st.drivers || []).map(function (x) {
        return h('tr', null, h('td', { class: 'r' }, x.pos), h('td', null, h('strong', null, x.code || ''), ' ', h('span', { class: 'muted' }, x.name)),
          h('td', null, x.team || ''), h('td', { class: 'r' }, x.wins), h('td', { class: 'r' }, num(x.points)));
      }));
    table($('#f1-std-cons'), [{ t: '#', r: 1 }, { t: 'Team' }, { t: 'Wins', r: 1 }, { t: 'Pts', r: 1 }],
      (st.constructors || []).map(function (x) {
        return h('tr', null, h('td', { class: 'r' }, x.pos), h('td', null, x.name), h('td', { class: 'r' }, x.wins), h('td', { class: 'r' }, num(x.points)));
      }));
    table($('#f1-calendar'), [{ t: 'Rd', r: 1 }, { t: 'Grand Prix' }, { t: 'Country' }, { t: 'Race start' }, { t: 'Winner' }],
      (d.calendar || []).map(function (x) {
        var isNext = n && x.round === n.round;
        return h('tr', { class: isNext ? 'win' : null }, h('td', { class: 'r' }, x.round),
          h('td', null, x.name, x.sprint ? h('span', { class: 'badge', style: 'margin-left:6px' }, 'Sprint') : null),
          h('td', { class: 'muted' }, x.country), h('td', { class: 'num' }, fmtFull(x.race_time)),
          h('td', null, x.winner || (isNext ? h('span', { class: 'muted' }, 'next') : '')));
      }));
  }

  // ---------------------------------------------------------- charts
  function calibrationChart(series) {
    var W = 460, H = 300, m = { l: 40, r: 12, t: 10, b: 34 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b;
    var X = function (v) { return m.l + v * iw; }, Y = function (v) { return m.t + (1 - v) * ih; };
    var svg = s('svg', { viewBox: '0 0 ' + W + ' ' + H, class: 'chart', role: 'img',
      'aria-label': 'Calibration: predicted probability versus observed win rate' });
    [0, .25, .5, .75, 1].forEach(function (v) {
      svg.appendChild(s('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: 'gridline' }));
      var ty = s('text', { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end' }); ty.textContent = pct(v); svg.appendChild(ty);
      var tx = s('text', { x: X(v), y: H - m.b + 16, 'text-anchor': 'middle' }); tx.textContent = pct(v); svg.appendChild(tx);
    });
    var xl = s('text', { x: m.l + iw / 2, y: H - 2, 'text-anchor': 'middle' }); xl.textContent = 'Predicted probability'; svg.appendChild(xl);
    svg.appendChild(s('line', { x1: X(0), y1: Y(0), x2: X(1), y2: Y(1), class: 'ref' }));
    series.forEach(function (sr) {
      if (!sr.bins || !sr.bins.length) return;
      var pts = sr.bins.map(function (b) { return X(b.mean_p) + ',' + Y(b.hit_rate); }).join(' ');
      svg.appendChild(s('polyline', { points: pts, fill: 'none', stroke: sr.color, 'stroke-width': 2, 'stroke-linejoin': 'round' }));
      sr.bins.forEach(function (b) {
        svg.appendChild(s('circle', { cx: X(b.mean_p), cy: Y(b.hit_rate), r: 4, fill: sr.color, stroke: 'var(--surface)', 'stroke-width': 2 }));
        var hit = s('circle', { cx: X(b.mean_p), cy: Y(b.hit_rate), r: 12, fill: 'transparent', tabindex: 0 });
        var show = function (e) {
          var r = hit.getBoundingClientRect();
          showTip(e.clientX ? e : { clientX: r.left + 12, clientY: r.top + 12 }, sr.name + ' · ' + b.n + ' forecasts',
            [{ color: sr.color, value: pct(b.mean_p), label: 'predicted' }, { value: pct(b.hit_rate), label: 'actually won' }]);
        };
        hit.addEventListener('pointermove', show); hit.addEventListener('focus', show);
        hit.addEventListener('pointerleave', hideTip); hit.addEventListener('blur', hideTip);
        svg.appendChild(hit);
      });
    });
    return svg;
  }

  function pnlChart(curve) {
    var W = 460, H = 300, m = { l: 46, r: 12, t: 10, b: 34 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b;
    var vals = curve.map(function (c) { return c.cum; }).concat([0]);
    var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
    var pad = Math.max((hi - lo) * 0.1, 0.5); lo -= pad; hi += pad;
    var n = curve.length;
    var X = function (i) { return m.l + (n <= 1 ? iw / 2 : i / (n - 1) * iw); };
    var Y = function (v) { return m.t + (hi - v) / (hi - lo) * ih; };
    var svg = s('svg', { viewBox: '0 0 ' + W + ' ' + H, class: 'chart', role: 'img', 'aria-label': 'Cumulative paper profit and loss in dollars' });
    var step = niceStep((hi - lo) / 4);
    for (var v = Math.ceil(lo / step) * step; v <= hi; v += step) {
      svg.appendChild(s('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: 'gridline' }));
      var t = s('text', { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end' }); t.textContent = (v < 0 ? '−$' : '$') + Math.abs(v).toFixed(step < 1 ? 1 : 0); svg.appendChild(t);
    }
    svg.appendChild(s('line', { x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0), class: 'axis' }));
    var xl = s('text', { x: m.l + iw / 2, y: H - 2, 'text-anchor': 'middle' }); xl.textContent = n + ' paper trades, in order'; svg.appendChild(xl);
    svg.appendChild(s('polyline', { points: curve.map(function (c, i) { return X(i) + ',' + Y(c.cum); }).join(' '),
      fill: 'none', stroke: 'var(--model)', 'stroke-width': 2, 'stroke-linejoin': 'round' }));
    var hair = s('line', { y1: m.t, y2: m.t + ih, class: 'hair', visibility: 'hidden' });
    var dot = s('circle', { r: 4, fill: 'var(--model)', stroke: 'var(--surface)', 'stroke-width': 2, visibility: 'hidden' });
    svg.appendChild(hair); svg.appendChild(dot);
    var hit = s('rect', { x: m.l, y: m.t, width: iw, height: ih, fill: 'transparent' });
    hit.addEventListener('pointermove', function (e) {
      var r = svg.getBoundingClientRect();
      var px = (e.clientX - r.left) / r.width * W;
      var i = Math.max(0, Math.min(n - 1, Math.round((px - m.l) / iw * (n - 1))));
      var c = curve[i];
      hair.setAttribute('x1', X(i)); hair.setAttribute('x2', X(i)); hair.setAttribute('visibility', 'visible');
      dot.setAttribute('cx', X(i)); dot.setAttribute('cy', Y(c.cum)); dot.setAttribute('visibility', 'visible');
      showTip(e, 'Trade ' + (i + 1) + ' · ' + fmtDate(c.t), [{ color: 'var(--model)', value: (c.cum >= 0 ? '+$' : '−$') + Math.abs(c.cum).toFixed(2), label: 'cumulative' }]);
    });
    hit.addEventListener('pointerleave', function () { hideTip(); hair.setAttribute('visibility', 'hidden'); dot.setAttribute('visibility', 'hidden'); });
    svg.appendChild(hit);
    return svg;
  }

  function niceStep(x) {
    var p = Math.pow(10, Math.floor(Math.log10(x || 1))), f = x / p;
    return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p;
  }

  // --------------------------------------------------- track record
  function renderRecord() {
    var sport = state.recordSport, d = state.data[sport];
    var box = clear($('#record-body'));
    document.querySelectorAll('#record-sport button').forEach(function (b) {
      b.setAttribute('aria-pressed', String(b.getAttribute('data-sport') === sport));
    });
    var L = d && d.ledger;
    if (!L || !L.settled_events) {
      box.appendChild(h('div', { class: 'empty', style: 'margin-top:16px' },
        'No settled forecasts yet. The scheduled job logs a forecast before every ' + (sport === 'lol' ? 'match' : 'race') +
        ' and scores it once the result is in.' + (L && L.pending_events ? ' ' + L.pending_events + ' events are waiting for results.' : '')));
      return;
    }
    var hh = L.head_to_head, P = L.paper || {};
    function tile(label, value, sub) { return h('div', { class: 'tile' }, h('div', { class: 'label' }, label), h('div', { class: 'value' }, value), sub ? h('div', { class: 'sub' }, sub) : null); }
    box.appendChild(h('div', { class: 'tiles' },
      tile('Settled events', L.settled_events, L.pending_events + ' pending'),
      hh ? tile('Log loss · model', num(hh.model.log_loss, 3), 'market ' + num(hh.market.log_loss, 3) + ' · lower is better') : tile('Log loss · model', num(L.model.log_loss, 3)),
      hh ? tile('Brier · model', num(hh.model.brier, 3), 'market ' + num(hh.market.brier, 3)) : null,
      tile('Paper P&L', (P.pnl >= 0 ? '+$' : '−$') + Math.abs(P.pnl || 0).toFixed(2), (P.trades || 0) + ' trades' + (P.roi != null ? ' · ROI ' + pct(P.roi, 1) : ''))));

    var charts = h('div', { class: 'charts' });
    box.appendChild(charts);
    if (L.calibration_model) {
      var series = [{ name: 'Model', color: 'var(--model)', bins: L.calibration_model },
                    { name: 'Kalshi', color: 'var(--market)', bins: L.calibration_market }];
      charts.appendChild(h('div', { class: 'chart-card' }, h('h3', null, 'Calibration'),
        h('p', { class: 'note' }, 'Dots on the dashed line mean "70% favourites won 70% of the time".'),
        h('p', { class: 'note' }, series.map(function (sr) { return h('span', { class: 'legend-inline' }, h('span', { class: 'key-line', style: 'background:' + sr.color }), sr.name); })),
        calibrationChart(series), calTable(series)));
    }
    if (P.curve && P.curve.length) {
      charts.appendChild(h('div', { class: 'chart-card' }, h('h3', null, 'Cumulative paper P&L ($1 contracts)'),
        h('p', { class: 'note' }, 'Hypothetical trades when model − price ≥ ' + Math.round((L.edge_min || 0.05) * 100) + '¢ after estimated fees.'),
        pnlChart(P.curve)));
    }
    if (P.recent && P.recent.length) {
      box.appendChild(h('h2', { class: 'section-title' }, 'Recent paper trades'));
      var t = h('table', { class: 'data' });
      box.appendChild(h('div', { class: 'table-wrap' }, t));
      table(t, [{ t: 'Start' }, { t: 'Event' }, { t: 'Side' }, { t: 'Price', r: 1 }, { t: 'Model', r: 1 }, { t: 'P&L', r: 1 }],
        P.recent.map(function (x) {
          return h('tr', null, h('td', { class: 'num' }, fmtDate(x.event_start)), h('td', null, x.event_name),
            h('td', null, x.side + ' ' + x.selection), h('td', { class: 'r' }, cents(x.price)), h('td', { class: 'r' }, pct(x.p_model)),
            h('td', { class: 'r' }, h('span', { class: x.pnl > 0 ? 'hit' : 'miss' }, (x.pnl >= 0 ? '+' : '−') + '$' + Math.abs(x.pnl).toFixed(2))));
        }));
    }
    var rec = d.ledger_recent || [];
    box.appendChild(h('h2', { class: 'section-title' }, 'Latest closing forecasts'));
    var t2 = h('table', { class: 'data' });
    box.appendChild(h('div', { class: 'table-wrap' }, t2));
    table(t2, [{ t: 'Start' }, { t: 'Event' }, { t: 'Selection' }, { t: 'Model', r: 1 }, { t: 'Kalshi', r: 1 }, { t: 'Result' }],
      rec.map(function (x) {
        return h('tr', null, h('td', { class: 'num' }, fmtFull(x.event_start)), h('td', null, x.event_name), h('td', null, x.selection),
          h('td', { class: 'r' }, pct(x.p_model)), h('td', { class: 'r' }, cents(x.mid)),
          h('td', null, x.outcome == null ? h('span', { class: 'muted' }, 'pending') : x.outcome ? h('span', { class: 'hit' }, 'won') : h('span', { class: 'muted' }, 'lost')));
      }));
  }

  function calTable(series) {
    var t = h('table', { class: 'data' });
    var rows = [];
    series.forEach(function (sr) {
      (sr.bins || []).forEach(function (b) {
        rows.push(h('tr', null, h('td', null, sr.name), h('td', { class: 'r' }, pct(b.lo) + '–' + pct(b.hi)),
          h('td', { class: 'r' }, pct(b.mean_p, 1)), h('td', { class: 'r' }, pct(b.hit_rate, 1)), h('td', { class: 'r' }, b.n)));
      });
    });
    table(t, [{ t: 'Series' }, { t: 'Bucket', r: 1 }, { t: 'Avg predicted', r: 1 }, { t: 'Won', r: 1 }, { t: 'n', r: 1 }], rows);
    return h('details', null, h('summary', { class: 'note', style: 'cursor:pointer;margin-top:8px' }, 'Show as table'), h('div', { class: 'table-wrap' }, t));
  }

  // ------------------------------------------------------------ method
  function renderMethod() {
    var box = clear($('#method-body'));
    var lol = state.data.lol, f1 = state.data.f1;
    if (lol && lol.backtest) {
      var b = lol.backtest, chosen = lol.model.chosen;
      box.appendChild(h('h2', { class: 'section-title', style: 'margin-top:4px' }, 'LoL walk-forward backtest'));
      box.appendChild(h('p', { class: 'note' }, b.n + ' matches since ' + fmt(b.from, { month: 'short', day: 'numeric', year: 'numeric' }) +
        ', each predicted by a model fitted only on earlier matches. Elo K = ' + lol.model.elo.k + ', off-season shrink = ' + lol.model.elo.regress + ' (tuned on data before this window).'));
      var t = h('table', { class: 'data' });
      box.appendChild(h('div', { class: 'table-wrap' }, t));
      table(t, [{ t: 'Model' }, { t: 'Matches', r: 1 }, { t: 'Log loss', r: 1 }, { t: 'Brier', r: 1 }, { t: 'Accuracy', r: 1 }, { t: 'Intl log loss', r: 1 }, { t: 'Bo1 / Bo3+ log loss', r: 1 }],
        Object.keys(b.metrics).map(function (k) {
          var mm = b.metrics[k];
          return h('tr', { class: k === chosen ? 'win' : null },
            h('td', null, k, k === chosen ? h('span', { class: 'badge solid', style: 'margin-left:6px' }, 'live') : null),
            h('td', { class: 'r' }, mm.all.n), h('td', { class: 'r' }, num(mm.all.log_loss, 4)), h('td', { class: 'r' }, num(mm.all.brier, 4)),
            h('td', { class: 'r' }, pct(mm.all.accuracy, 1)), h('td', { class: 'r' }, num(mm.international.log_loss, 4) + ' (n=' + (mm.international.n || 0) + ')'),
            h('td', { class: 'r' }, num(mm.bo1.log_loss, 3) + ' / ' + num(mm['bo3+'].log_loss, 3)));
        }), 'A coin flip scores log loss 0.693 and Brier 0.25.');
    }
    if (f1 && f1.backtest) {
      var fb = f1.backtest;
      box.appendChild(h('h2', { class: 'section-title' }, 'F1 walk-forward backtest (' + fb.from_season + '–present)'));
      var t2 = h('table', { class: 'data' });
      box.appendChild(h('div', { class: 'table-wrap' }, t2));
      var rows = [];
      Object.keys(fb.metrics).sort().reverse().forEach(function (stage) {
        Object.keys(fb.metrics[stage]).forEach(function (k) {
          var mm = fb.metrics[stage][k], live = fb.chosen[stage] === k;
          rows.push(h('tr', { class: live ? 'win' : null }, h('td', null, stage === 'post_quali' ? 'After quali' : 'Before quali'),
            h('td', null, k, live ? h('span', { class: 'badge solid', style: 'margin-left:6px' }, 'live') : null),
            h('td', { class: 'r' }, mm.races), h('td', { class: 'r' }, num(mm.winner_log_loss, 3)),
            h('td', { class: 'r' }, num(mm.win_brier, 4)), h('td', { class: 'r' }, pct(mm.favourite_won, 1))));
        });
      });
      table(t2, [{ t: 'Stage' }, { t: 'Model' }, { t: 'Races', r: 1 }, { t: 'Winner log loss', r: 1 }, { t: 'Win Brier', r: 1 }, { t: 'Favourite won', r: 1 }], rows,
        'Winner log loss = average −log(probability given to the actual winner). Guessing uniformly over 20 drivers scores 3.0.');
      var coefs = f1.next_race && f1.next_race.coefficients;
      if (coefs) {
        box.appendChild(h('p', { class: 'note', style: 'margin-top:10px' }, 'Live F1 model weights (standardised features; positive raises win chance): ' +
          Object.keys(coefs).map(function (k) { return k + ' ' + signed(coefs[k], 2); }).join(' · ')));
      }
    }
    var m = state.data.meta;
    if (m) $('#edge-min').textContent = Math.round((m.edge_min || 0.05) * 100);
  }

  // ------------------------------------------------------ chrome & boot
  function setupTimezones() {
    var sel = $('#tz-select');
    var local = Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';
    var zones = [];
    try { zones = Intl.supportedValuesOf('timeZone'); } catch (e) { zones = [local, 'UTC']; }
    if (zones.indexOf('UTC') < 0) zones.unshift('UTC');
    var saved = null;
    try { saved = localStorage.getItem('f1lol-tz'); } catch (e) { /* ignore */ }
    state.tz = saved && zones.indexOf(saved) >= 0 ? saved : local;
    zones.forEach(function (z) { sel.appendChild(h('option', { value: z, selected: z === state.tz }, z.replace(/_/g, ' ') + (z === local ? ' (yours)' : ''))); });
    sel.addEventListener('change', function () {
      state.tz = sel.value;
      try { localStorage.setItem('f1lol-tz', sel.value); } catch (e) { /* ignore */ }
      renderAll();
    });
  }

  function setupTheme() {
    var cur = 'auto';
    try { cur = localStorage.getItem('f1lol-theme') || 'auto'; } catch (e) { /* ignore */ }
    var btns = document.querySelectorAll('#theme-toggle button');
    function apply(v) {
      if (v === 'auto') document.documentElement.removeAttribute('data-theme');
      else document.documentElement.setAttribute('data-theme', v);
      btns.forEach(function (b) { b.setAttribute('aria-pressed', String(b.getAttribute('data-theme-choice') === v)); });
      try { localStorage.setItem('f1lol-theme', v); } catch (e) { /* ignore */ }
    }
    btns.forEach(function (b) { b.addEventListener('click', function () { apply(b.getAttribute('data-theme-choice')); }); });
    apply(cur);
  }

  function setupTabs() {
    var tabs = document.querySelectorAll('.tabs button');
    function show(name) {
      tabs.forEach(function (t) {
        var on = t.getAttribute('data-tab') === name;
        t.setAttribute('aria-selected', String(on));
        t.tabIndex = on ? 0 : -1;
        document.getElementById('panel-' + t.getAttribute('data-tab')).hidden = !on;
      });
    }
    tabs.forEach(function (t) {
      t.addEventListener('click', function () { history.replaceState(null, '', '#' + t.getAttribute('data-tab')); show(t.getAttribute('data-tab')); });
    });
    var initial = (location.hash || '#lol').slice(1);
    show(document.getElementById('panel-' + initial) ? initial : 'lol');
    document.querySelectorAll('#record-sport button').forEach(function (b) {
      b.addEventListener('click', function () { state.recordSport = b.getAttribute('data-sport'); renderRecord(); });
    });
    $('#lol-search').addEventListener('input', applyLoLFilters);
    $('#lol-kalshi-only').addEventListener('change', applyLoLFilters);
  }

  function renderMeta() {
    var m = state.data.meta;
    var line = clear($('#meta-line'));
    if (!m) { line.textContent = 'No data yet — run the pipeline.'; return; }
    var bad = Object.keys(m.status || {}).filter(function (k) { return m.status[k] !== 'ok'; });
    line.appendChild(document.createTextNode('Updated ' + ago(m.updated_at) + ' (' + fmtFull(m.updated_at) + ') · model ' + m.model_version));
    if (bad.length) line.appendChild(h('span', { class: 'miss' }, ' · last update failed for ' + bad.join(', ') + ' (showing previous data)'));
    $('#footer-updated').textContent = 'data refreshed ' + fmtFull(m.updated_at);
  }

  function renderAll() {
    renderMeta();
    try { renderLoL(); } catch (e) { console.error(e); }
    try { renderF1(); } catch (e) { console.error(e); }
    try { renderRecord(); } catch (e) { console.error(e); }
    try { renderMethod(); } catch (e) { console.error(e); }
  }

  function load(name, bust) {
    return fetch('data/' + name + '.json' + (bust ? '?v=' + encodeURIComponent(bust) : ''), { cache: 'no-cache' })
      .then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; });
  }

  function boot() {
    var gh = $('#gh-link');
    var host = location.hostname, owner = host.split('.')[0];
    var repo = location.pathname.split('/').filter(Boolean)[0];
    if (/github\.io$/.test(host)) {
      gh.href = 'https://github.com/' + owner + '/' + (repo || host);
      $('#ledger-link').href = gh.href + '/blob/main/ledger/predictions.csv';
    } else { gh.hidden = true; $('#ledger-link').href = 'ledger/predictions.csv'; }
    setupTheme(); setupTimezones(); setupTabs();
    load('meta').then(function (meta) {
      state.data.meta = meta;
      var v = meta && meta.updated_at;
      return Promise.all([load('lol', v), load('f1', v)]);
    }).then(function (r) {
      state.data.lol = r[0]; state.data.f1 = r[1];
      renderAll();
    });
    setInterval(function () {
      document.querySelectorAll('[data-until]').forEach(function (el) { el.textContent = until(el.getAttribute('data-until')); });
    }, 30000);
  }

  boot();
})();
