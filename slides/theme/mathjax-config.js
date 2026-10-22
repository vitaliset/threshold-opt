// MathJax reads the same TeX as the Beamer deck, with $...$ for inline math.
// Typesetting is started by deck.js, before reveal.js lays the slides out.
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['\\[', '\\]']],
    processEscapes: true
  },
  svg: { fontCache: 'local', scale: 0.95 },
  options: { skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] },
  startup: { typeset: false }
};
