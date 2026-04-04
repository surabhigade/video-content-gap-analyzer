/* ============================================================
   Video Content Gap Analyzer — Interactions
   ============================================================ */

(function () {
  'use strict';

  /* ---- Theme Toggle ---- */
  var THEME_KEY = 'vcga-theme';
  var html = document.documentElement;

  function getPreferred() {
    var stored = localStorage.getItem(THEME_KEY);
    if (stored) return stored;
    return 'light'; // default to light
  }

  function applyTheme(theme) {
    if (theme === 'dark') {
      html.setAttribute('data-theme', 'dark');
    } else {
      html.removeAttribute('data-theme');
    }
    localStorage.setItem(THEME_KEY, theme);
  }

  // Apply immediately (before DOMContentLoaded to prevent flash)
  applyTheme(getPreferred());

  /* ---- Scroll Reveal (IntersectionObserver) ---- */
  var STAGGER_MS = 100;

  var observer = new IntersectionObserver(
    function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;

        var el = entry.target;
        el.classList.add('visible');

        if (el.classList.contains('reveal-group')) {
          var children = el.querySelectorAll('.reveal-child');
          children.forEach(function (child, i) {
            child.style.setProperty('--delay', i * STAGGER_MS + 'ms');
          });

          // Animate pipeline connectors sequentially after steps appear
          var steps = el.querySelectorAll('.pipeline-step');
          steps.forEach(function (step, i) {
            step.style.setProperty('--connector-delay', (i * STAGGER_MS + 200) + 'ms');
          });
        }

        observer.unobserve(el);
      });
    },
    { threshold: 0.15 }
  );

  /* ---- DOMContentLoaded ---- */
  document.addEventListener('DOMContentLoaded', function () {

    // Observe reveal elements
    var targets = document.querySelectorAll('.reveal, .reveal-group');
    targets.forEach(function (el) {
      observer.observe(el);
    });

    // Theme toggle button
    var toggle = document.getElementById('theme-toggle');
    if (toggle) {
      toggle.addEventListener('click', function () {
        var current = html.hasAttribute('data-theme') ? 'dark' : 'light';
        applyTheme(current === 'dark' ? 'light' : 'dark');
      });
    }

    // Navbar scroll effect
    var nav = document.querySelector('.nav');
    var backToTop = document.querySelector('.back-to-top');
    var scrollThreshold = 80;
    var topThreshold = 400;

    function onScroll() {
      var y = window.scrollY;

      // Nav background
      if (y > scrollThreshold) {
        nav.classList.add('scrolled');
      } else {
        nav.classList.remove('scrolled');
      }

      // Back to top visibility
      if (backToTop) {
        if (y > topThreshold) {
          backToTop.classList.add('visible');
        } else {
          backToTop.classList.remove('visible');
        }
      }
    }

    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll(); // initial check

    // Back to top click
    if (backToTop) {
      backToTop.addEventListener('click', function () {
        window.scrollTo({ top: 0, behavior: 'smooth' });
      });
    }

    /* ---- Mobile Hamburger Toggle ---- */
    var hamburger = document.querySelector('.nav-hamburger');
    var navLinksContainer = document.querySelector('.nav-links');

    if (hamburger && navLinksContainer) {
      hamburger.addEventListener('click', function () {
        var isOpen = hamburger.classList.toggle('open');
        navLinksContainer.classList.toggle('open');
        hamburger.setAttribute('aria-expanded', isOpen);
      });

      // Close menu when a nav link is clicked
      navLinksContainer.querySelectorAll('.nav-link').forEach(function (link) {
        link.addEventListener('click', function () {
          hamburger.classList.remove('open');
          navLinksContainer.classList.remove('open');
          hamburger.setAttribute('aria-expanded', 'false');
        });
      });
    }

    /* ---- Scroll Spy for Nav Links ---- */
    var navLinks = document.querySelectorAll('.nav-link');
    var sectionIds = [];
    navLinks.forEach(function (link) {
      var id = link.getAttribute('href').replace('#', '');
      sectionIds.push(id);
    });

    if (navLinks.length > 0) {
      var spyObserver = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            var id = entry.target.id;
            navLinks.forEach(function (link) {
              link.classList.toggle('active', link.getAttribute('href') === '#' + id);
            });
          }
        });
      }, {
        rootMargin: '-20% 0px -60% 0px',
        threshold: 0
      });

      sectionIds.forEach(function (id) {
        var section = document.getElementById(id);
        if (section) spyObserver.observe(section);
      });
    }

    /* ---- Smooth Scroll with Offset for Fixed Navbar ---- */
    document.querySelectorAll('a[href^="#"]').forEach(function (anchor) {
      anchor.addEventListener('click', function (e) {
        var targetId = this.getAttribute('href');
        if (targetId === '#') return;
        var target = document.querySelector(targetId);
        if (target) {
          e.preventDefault();
          var offset = 80; // 56px nav + 24px padding
          var top = target.getBoundingClientRect().top + window.scrollY - offset;
          window.scrollTo({ top: top, behavior: 'smooth' });
        }
      });
    });

    /* ---- Coverage Score Counter Animation ---- */
    var coverageEl = document.querySelector('.coverage-counter');
    if (coverageEl) {
      var counterObserver = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) return;
          var target = parseInt(coverageEl.getAttribute('data-target'), 10);
          var startTime = null;
          var duration = 1500;

          function step(timestamp) {
            if (!startTime) startTime = timestamp;
            var progress = Math.min((timestamp - startTime) / duration, 1);
            // Ease-out cubic
            var eased = 1 - Math.pow(1 - progress, 3);
            var current = Math.round(eased * target);
            coverageEl.textContent = current + '%';
            if (progress < 1) {
              requestAnimationFrame(step);
            }
          }

          requestAnimationFrame(step);
          counterObserver.unobserve(entry.target);
        });
      }, { threshold: 0.5 });

      counterObserver.observe(coverageEl);
    }

    /* ---- Button Ripple Effect ---- */
    document.querySelectorAll('.btn').forEach(function (btn) {
      function updateRipple(e) {
        var rect = btn.getBoundingClientRect();
        var x = ((e.clientX - rect.left) / rect.width) * 100;
        var y = ((e.clientY - rect.top) / rect.height) * 100;
        btn.style.setProperty('--ripple-x', x + '%');
        btn.style.setProperty('--ripple-y', y + '%');
      }

      btn.addEventListener('mouseenter', updateRipple);
      btn.addEventListener('mousemove', updateRipple);
    });
  });
})();
