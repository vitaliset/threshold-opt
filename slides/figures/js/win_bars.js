// Win-or-tie rates as horizontal bars, for the four result slides. The
// numbers come from figures/data/win_rates.js (slides/tools/export_data.py):
// the default view, all cells, equals the static chart; the menu restricts
// the cells to one base classifier or one objective. Whiskers are 95%
// bootstrap intervals over cells. Hover a bar for its exact values.
(function () {
  'use strict';
  var fx = Deck.fx;

  // Short names of the base classifiers, as on the slides.
  var SHORT = { 'LGBMClassifier()': 'LGBM', 'XGBoost': 'XGB', 'CatBoost': 'Cat', 'RandomForest': 'RF', 'ExtraTrees': 'ET',
    'LogisticRegression()': 'LogReg', 'GaussianNB': 'NB' };

  function chart(key) {
    return function (host, state) {
      var data = Deck.data.win_rates[key], families = Deck.data.win_rates.families;
      var slice = data.slices[state.preset] ? state.preset : 'all';
      // The binary charts can add the calibrate-then-argmax strategies.
      var base = data.shown, extra = key.indexOf('binary') !== 0 ? [] : Object.keys(data.family).filter(function (s) {
        return base.indexOf(s) < 0 && data.family[s] === 'argmax' && s !== 'Argmax';
      });
      var row0 = base.length > 8 ? 32 : 36;   // the nine-row multiclass charts are a little tighter
      var rows = base, rh = row0, L = 520, R = 1120, T = 8, more = false;
      var W = 1220, H = T + base.length * row0 + 72;   // the height stays fixed when rows are added
      fx.flex(host, 'column', 2);
      var top = fx.html('div', 'ctl', host);
      top.style.marginTop = '0';
      var used = {};
      rows.forEach(function (s) { used[data.family[s]] = true; });
      ['cv', 'split', 'analytical', 'argmax'].forEach(function (f) {
        if (!used[f]) return;
        var item = fx.html('span', '', top);
        item.innerHTML = '<span class="swatch" style="height:16px;width:22px;background:' + families[f].color + '"></span>' + families[f].label;
      });
      var label = fx.html('label', '', top);
      fx.html('span', '', label, 'cells:');
      var select = fx.html('select', '', label);
      var add = function (parent, value, text) {
        var o = fx.html('option', '', parent, text); o.value = value; return o;
      };
      add(select, 'all', 'all (' + data.cells + ')');
      var gm = fx.html('optgroup', '', select); gm.label = 'one base classifier';
      data.models.forEach(function (m) { add(gm, 'model:' + m, SHORT[m] || m); });
      var go = fx.html('optgroup', '', select); go.label = 'one objective';
      data.objectives.forEach(function (o) { add(go, 'objective:' + o, o); });
      select.value = slice;
      select.addEventListener('change', function () { slice = select.value; draw(); });
      if (extra.length) fx.button(top, '+ calibrated argmax', function (b) {
        more = !more;
        b.classList.toggle('on', more);
        rows = more ? base.concat(extra) : base;
        rh = base.length * row0 / rows.length;
        draw();
      });
      var svg = fx.svg(host, W, H);
      var layer = fx.el('g', {}, svg);
      var X = function (v) { return L + v / 100 * (R - L); };

      function draw() {
        fx.clear(layer);
        var values = data.slices[slice];
        [0, 25, 50, 75, 100].forEach(function (t) {
          fx.el('line', { x1: X(t), y1: T, x2: X(t), y2: T + rows.length * rh, stroke: '#dcdcdc', 'stroke-width': 1.5 }, layer);
          fx.text(layer, X(t), T + rows.length * rh + 28, String(t), { 'font-size': 26 });
        });
        fx.text(layer, (L + R) / 2, T + rows.length * rh + 62, 'cells won or tied (%)', { 'font-size': 26 });
        rows.forEach(function (s, i) {
          var y = T + i * rh + rh / 2, v = values[s];
          var fs = rh < 30 ? 21 : 27;
          fx.text(layer, L - 12, y + fs / 3, s, { 'font-size': fs, 'text-anchor': 'end' });
          if (!v) { fx.text(layer, X(0) + 10, y + 8, 'no results', { 'font-size': 20, 'text-anchor': 'start', fill: fx.grey }); return; }
          var g = fx.el('g', {}, layer);
          fx.el('rect', { x: X(0), y: y - rh / 3, width: Math.max(X(v[0]) - X(0), 0.5), height: rh * 2 / 3, fill: families[data.family[s]].color }, g);
          var lo = Math.max(v[0] - v[1], 0), hi = Math.min(v[0] + v[1], 100);
          fx.el('path', { d: 'M' + X(lo) + ' ' + y + ' L' + X(hi) + ' ' + y + ' M' + X(lo) + ' ' + (y - 6) + ' L' + X(lo) + ' ' + (y + 6) + ' M' + X(hi) + ' ' + (y - 6) + ' L' + X(hi) + ' ' + (y + 6), stroke: '#555', 'stroke-width': 2, fill: 'none' }, g);
          fx.text(layer, X(hi) + 12, y + fs / 3, v[0] > 99.5 && v[0] < 100 ? fx.fmt(v[0], 1) : fx.fmt(v[0], 0), { 'font-size': fs, 'text-anchor': 'start' });
          fx.el('title', {}, g).textContent = s + ': ' + fx.fmt(v[0], 1) + '% ± ' + fx.fmt(v[1], 1) + ' of ' + v[2] + ' cells';
        });
        fx.el('line', { x1: X(0), y1: T, x2: X(0), y2: T + rows.length * rh, stroke: '#999', 'stroke-width': 2 }, layer);
      }
      draw();
    };
  }

  ['binary_cost', 'binary_max', 'multiclass_cost', 'multiclass_max'].forEach(function (key) {
    Deck.figures['win_bars_' + key] = chart(key);
  });
})();
