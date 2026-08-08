// Theme toggle + sidebar filter for the multi-page docs.
(function () {
  var root = document.documentElement;
  var btn = document.getElementById('theme-btn');
  if (btn) btn.addEventListener('click', function () {
    var cur = root.getAttribute('data-theme');
    var next = cur === 'dark' ? 'light' : (cur === 'light' ? 'dark' :
      (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'light' : 'dark'));
    root.setAttribute('data-theme', next);
  });
  var input = document.getElementById('search');
  if (input) {
    var links = Array.prototype.slice.call(document.querySelectorAll('#nav a'));
    input.addEventListener('input', function () {
      var q = input.value.toLowerCase().trim();
      links.forEach(function (a) {
        var hit = !q || a.textContent.toLowerCase().indexOf(q) !== -1;
        a.classList.toggle('hide', !hit);
      });
      document.querySelectorAll('#nav .group').forEach(function (g) {
        g.classList.toggle('hide', !!q);
      });
    });
  }
})();
