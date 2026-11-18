// Shared helpers of the interactive figures: SVG drawing, the K = 3 simplex,
// regions of an argmin of affine scores, flat controls, math labels.
//
// A figure registers itself as Deck.figures[name] = function (host, state).
// `host` is an empty div inside the slide's <figure>; `state` comes from the
// figure's data-state attribute and may carry a `preset` name (also settable
// as ?name=preset in the URL, which the check uses).
(function () {
  'use strict';

  var Deck = window.Deck = window.Deck || {};
  Deck.figures = Deck.figures || {};
  var NS = 'http://www.w3.org/2000/svg';

  var fx = Deck.fx = {
    // Colours of the deck; classes 0, 1, 2 are green, blue, red as in the thesis figures.
    ink: '#232830', grey: '#666666', rule: '#D9D9D9',
    blue: '#142D69', yellow: '#E6AF3C', red: '#82143C', softblue: '#0064A0',
    cls: ['#2E8B3D', '#2C3FC7', '#D22B2B'],
    fill: ['#D9F2D9', '#D9D9FF', '#FFD9D9']
  };

  fx.el = function (tag, attrs, parent) {
    var node = document.createElementNS(NS, tag);
    for (var k in attrs || {}) node.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(node);
    return node;
  };

  fx.html = function (tag, cls, parent, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  };

  // An SVG of w x h stage pixels inside `parent`.
  fx.svg = function (parent, w, h) {
    return fx.el('svg', { 'class': 'chart', viewBox: '0 0 ' + w + ' ' + h, width: w, height: h }, parent);
  };

  fx.clear = function (node) { while (node.firstChild) node.removeChild(node.firstChild); };

  fx.text = function (parent, x, y, str, attrs) {
    var t = fx.el('text', Object.assign({ x: x, y: y, 'font-size': 24, fill: fx.ink, 'text-anchor': 'middle' }, attrs || {}), parent);
    t.textContent = str;
    return t;
  };

  // A TeX label typeset by MathJax, placed with its anchor point at (x, y).
  // anchor: 'middle' | 'start' | 'end'; the label is centred vertically on y.
  fx.tex = function (parent, x, y, tex, opts) {
    opts = opts || {};
    var size = opts.size || 26;
    var svg = MathJax.tex2svg(tex, { display: false }).querySelector('svg');
    var ex = size * 0.45;
    var w = parseFloat(svg.getAttribute('width')) * ex;
    var h = parseFloat(svg.getAttribute('height')) * ex;
    var ax = opts.anchor === 'start' ? 0 : opts.anchor === 'end' ? w : w / 2;
    svg.setAttribute('x', x - ax);
    svg.setAttribute('y', y - h / 2);
    svg.setAttribute('width', w);
    svg.setAttribute('height', h);
    svg.removeAttribute('style');
    svg.style.color = opts.color || fx.ink;
    svg.style.pointerEvents = 'none';
    parent.appendChild(svg);
    return svg;
  };

  fx.fmt = function (v, digits) {
    return Number(v).toFixed(digits == null ? 2 : digits);
  };

  // ---- controls ----

  // A labelled slider with a numeric readout. opts: label (HTML), min, max,
  // step, value, digits, format(v). Returns the input; input.set(v) moves it.
  fx.slider = function (parent, opts, onInput) {
    var label = fx.html('label', '', parent);
    var name = fx.html('span', '', label);
    name.innerHTML = opts.label;
    var input = fx.html('input', '', label);
    input.type = 'range';
    input.min = opts.min; input.max = opts.max; input.step = opts.step; input.value = opts.value;
    if (opts.width) input.style.width = opts.width + 'px';
    var out = fx.html('output', '', label);
    var show = function () {
      out.textContent = opts.format ? opts.format(Number(input.value)) : fx.fmt(input.value, opts.digits);
    };
    input.addEventListener('input', function () { show(); onInput(Number(input.value)); });
    input.set = function (v) { input.value = v; show(); };
    show();
    return input;
  };

  fx.button = function (parent, text, onClick) {
    var b = fx.html('button', '', parent, text);
    b.type = 'button';
    b.addEventListener('click', function () { onClick(b); });
    return b;
  };

  // A row of buttons of which one is active. choices: [[key, text], ...].
  fx.choice = function (parent, choices, current, onPick) {
    var buttons = {};
    var mark = function (key) { for (var k in buttons) buttons[k].classList.toggle('on', k === key); };
    choices.forEach(function (c) {
      buttons[c[0]] = fx.button(parent, c[1], function () { mark(c[0]); onPick(c[0]); });
    });
    mark(current);
    return { mark: mark };
  };

  // Make an SVG node draggable; onMove gets the pointer in SVG coordinates.
  fx.drag = function (node, svg, onMove) {
    node.classList.add('drag');
    var move = function (ev) {
      var pt = svg.createSVGPoint();
      pt.x = ev.clientX; pt.y = ev.clientY;
      var at = pt.matrixTransform(svg.getScreenCTM().inverse());
      onMove(at.x, at.y);
    };
    node.addEventListener('pointerdown', function (ev) {
      ev.preventDefault(); ev.stopPropagation();
      node.setPointerCapture(ev.pointerId);
      node.addEventListener('pointermove', move);
    });
    var stop = function (ev) { node.removeEventListener('pointermove', move); };
    node.addEventListener('pointerup', stop);
    node.addEventListener('pointercancel', stop);
  };

  // ---- the simplex for K = 3 ----

  // e0 at the top, e1 bottom right, e2 bottom left (as in the thesis figures).
  fx.simplex = function (cx, top, side) {
    var hgt = side * Math.sqrt(3) / 2;
    var v = [[cx, top], [cx + side / 2, top + hgt], [cx - side / 2, top + hgt]];
    var S = {
      v: v, side: side, height: hgt,
      xy: function (p) {
        return [p[0] * v[0][0] + p[1] * v[1][0] + p[2] * v[2][0],
                p[0] * v[0][1] + p[1] * v[1][1] + p[2] * v[2][1]];
      },
      bary: function (x, y) {
        var p0 = (v[1][1] - y) / hgt;
        var p1 = (x - v[2][0]) / side - p0 / 2;
        return [p0, p1, 1 - p0 - p1];
      },
      // Nearest point of the simplex (Euclidean projection).
      clamp: function (p) {
        var u = p.slice().sort(function (a, b) { return b - a; });
        var css = 0, rho = 0, theta = 0;
        for (var i = 0; i < 3; i++) {
          css += u[i];
          if (u[i] - (css - 1) / (i + 1) > 0) { rho = i + 1; theta = (css - 1) / rho; }
        }
        return p.map(function (x) { return Math.max(x - theta, 0); });
      },
      path: function (poly) {
        return poly.map(function (p, i) { var q = S.xy(p); return (i ? 'L' : 'M') + q[0].toFixed(2) + ' ' + q[1].toFixed(2); }).join(' ') + ' Z';
      },
      outline: function (parent) {
        return fx.el('path', { d: S.path([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), fill: 'none', stroke: fx.ink, 'stroke-width': 3, 'stroke-linejoin': 'round' }, parent);
      },
      // Vertex labels e_0, e_1, e_2 outside the triangle.
      labels: function (parent, size) {
        var off = [[0, -22], [34, 8], [-34, 8]];
        for (var k = 0; k < 3; k++) fx.tex(parent, v[k][0] + off[k][0], v[k][1] + off[k][1], '\\mathbf{e}_' + k, { size: size || 26 });
      }
    };
    return S;
  };

  // The part of polygon `poly` (barycentric points) where f(p) <= 0, f affine.
  fx.clip = function (poly, f) {
    var out = [];
    for (var i = 0; i < poly.length; i++) {
      var a = poly[i], b = poly[(i + 1) % poly.length];
      var fa = f(a), fb = f(b);
      if (fa <= 0) out.push(a);
      if ((fa < 0 && fb > 0) || (fa > 0 && fb < 0)) {
        var t = fa / (fa - fb);
        out.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])]);
      }
    }
    return out;
  };

  // Regions of p -> argmin_k score_k(p) on the simplex, for affine scores
  // given as functions; ties go to the smallest index, as in the Bayes rule.
  fx.regions = function (scores) {
    var tri = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    return scores.map(function (si, i) {
      var poly = tri;
      scores.forEach(function (sk, k) {
        if (k !== i && poly.length) poly = fx.clip(poly, function (p) { return si(p) - sk(p); });
      });
      return poly;
    });
  };

  fx.area = function (poly) {
    var a = 0;
    for (var i = 0; i < poly.length; i++) {
      var p = poly[i], q = poly[(i + 1) % poly.length];
      a += p[0] * q[1] - q[0] * p[1];
    }
    return Math.abs(a) / 2;
  };

  fx.centroid = function (poly) {
    var c = [0, 0, 0];
    poly.forEach(function (p) { for (var k = 0; k < 3; k++) c[k] += p[k] / poly.length; });
    return c;
  };

  fx.argmin = function (values) {
    var best = 0;
    for (var k = 1; k < values.length; k++) if (values[k] < values[best]) best = k;
    return best;
  };

  // Draw filled regions with the boundaries between them. Returns the group.
  fx.drawRegions = function (parent, S, regions, opts) {
    opts = opts || {};
    var g = fx.el('g', {}, parent);
    regions.forEach(function (poly, k) {
      if (poly.length < 3) return;
      fx.el('path', { d: S.path(poly), fill: (opts.fill || fx.fill)[k], stroke: opts.stroke || '#8a8a8a', 'stroke-width': opts.width || 2.5, 'stroke-linejoin': 'round' }, g);
    });
    return g;
  };

  // Lay a node out as a flex row or column.
  fx.flex = function (node, dir, gap) {
    node.style.display = 'flex';
    node.style.flexDirection = dir || 'row';
    node.style.alignItems = 'center';
    node.style.justifyContent = 'center';
    node.style.gap = (gap == null ? 40 : gap) + 'px';
    return node;
  };

  // Set HTML that contains $...$ and typeset it.
  fx.math = function (node, html) {
    node.innerHTML = html;
    node.style.whiteSpace = 'nowrap';
    MathJax.typesetClear([node]);
    MathJax.typeset([node]);
    return node;
  };

  // Affine scores of the Voronoi rule with prototypes mus: the squared
  // distance to mu_k minus |p|^2, which is common to all k.
  fx.voronoi = function (mus) {
    return mus.map(function (m) {
      var sq = m[0] * m[0] + m[1] * m[1] + m[2] * m[2];
      return function (p) { return sq - 2 * (m[0] * p[0] + m[1] * p[1] + m[2] * p[2]); };
    });
  };

  // An arrow from (x, y) along (dx, dy), of the given length in pixels.
  fx.arrow = function (parent, x, y, dx, dy, len, color) {
    var n = Math.hypot(dx, dy) || 1, ux = dx / n, uy = dy / n;
    var ex = x + ux * len, ey = y + uy * len;
    fx.el('line', { x1: x, y1: y, x2: ex - ux * 8, y2: ey - uy * 8, stroke: color, 'stroke-width': 3 }, parent);
    fx.el('path', { d: 'M' + ex + ' ' + ey + ' L' + (ex - ux * 16 - uy * 7) + ' ' + (ey - uy * 16 + ux * 7) + ' L' + (ex - ux * 16 + uy * 7) + ' ' + (ey - uy * 16 - ux * 7) + ' Z', fill: color }, parent);
    return [ex, ey];
  };

  // Deterministic pseudo-random numbers (mulberry32), so a figure looks the
  // same on every load and in the check.
  fx.rng = function (seed) {
    var a = seed >>> 0;
    var next = function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
    next.normal = function () {
      var u = Math.max(next(), 1e-12), v = next();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };
    return next;
  };
})();
