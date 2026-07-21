(() => {
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  document.documentElement.classList.add("js");

  const revealItems = document.querySelectorAll(
    ".hero > *, .toc-card, .card, .doc section, .figure, .result-card, .terminal"
  );

  revealItems.forEach((el, index) => {
    el.classList.add("reveal");
    el.style.setProperty("--reveal-delay", `${Math.min(index * 24, 220)}ms`);
  });

  if (reduceMotion) {
    revealItems.forEach((el) => el.classList.add("is-visible"));
  } else {
    const revealObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          entry.target.classList.add("is-visible");
          revealObserver.unobserve(entry.target);
        });
      },
      { threshold: 0.08, rootMargin: "0px 0px -8% 0px" }
    );

    revealItems.forEach((el) => revealObserver.observe(el));
  }

  const sideToc = document.querySelector(".side-toc");
  if (!sideToc) return;

  const indicator = document.createElement("span");
  indicator.className = "side-toc-indicator";
  indicator.setAttribute("aria-hidden", "true");
  sideToc.appendChild(indicator);

  const links = [...sideToc.querySelectorAll('a[href^="#"]')];
  const sections = links
    .map((link) => {
      const id = decodeURIComponent(link.getAttribute("href").slice(1));
      return { link, section: document.getElementById(id) };
    })
    .filter((item) => item.section);

  if (!sections.length) return;

  let suppressScrollspy = false;
  let suppressTimer = 0;

  const updateIndicator = (activeLink) => {
    const tocBox = sideToc.getBoundingClientRect();
    const linkBox = activeLink.getBoundingClientRect();
    sideToc.style.setProperty("--toc-indicator-top", `${linkBox.top - tocBox.top}px`);
    sideToc.style.setProperty("--toc-indicator-height", `${linkBox.height}px`);
    sideToc.classList.add("has-active");
  };

  const setActive = (activeLink, options = {}) => {
    if (!activeLink) return;

    if (options.jump) {
      sideToc.classList.add("is-jump");
    }

    links.forEach((link) => {
      const active = link === activeLink;
      link.classList.toggle("is-active", active);
      if (active) {
        link.setAttribute("aria-current", "true");
      } else {
        link.removeAttribute("aria-current");
      }
    });

    updateIndicator(activeLink);

    if (options.jump) {
      requestAnimationFrame(() => {
        requestAnimationFrame(() => sideToc.classList.remove("is-jump"));
      });
    }
  };

  setActive(sections[0].link);

  const findLinkForSection = (section) =>
    sections.find((item) => item.section === section)?.link;

  const isAtPageBottom = () =>
    window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 3;

  const updateActiveFromScrollPosition = () => {
    if (suppressScrollspy) return;

    if (isAtPageBottom()) {
      setActive(sections[sections.length - 1].link);
      return;
    }

    const marker = Math.min(140, window.innerHeight * 0.28);
    const current =
      [...sections]
        .reverse()
        .find(({ section }) => section.getBoundingClientRect().top <= marker) || sections[0];

    setActive(current.link);
  };

  const releaseScrollspyWhenArrived = (section) => {
    window.clearTimeout(suppressTimer);
    const startedAt = performance.now();

    const tick = () => {
      const top = section.getBoundingClientRect().top;
      const arrived = Math.abs(top - 92) < 12 || Math.abs(top) < 12;
      const timedOut = performance.now() - startedAt > 1400;

      if (arrived || timedOut) {
        suppressScrollspy = false;
        setActive(findLinkForSection(section));
        return;
      }

      suppressTimer = window.setTimeout(tick, 80);
    };

    tick();
  };

  links.forEach((link) => {
    link.addEventListener("click", (event) => {
      const item = sections.find(({ link: candidate }) => candidate === link);
      if (!item) return;

      event.preventDefault();
      suppressScrollspy = true;
      setActive(link, { jump: true });
      history.pushState(null, "", link.getAttribute("href"));

      item.section.scrollIntoView({
        behavior: reduceMotion ? "auto" : "smooth",
        block: "start",
      });

      releaseScrollspyWhenArrived(item.section);
    });
  });

  const sectionObserver = new IntersectionObserver(
    (entries) => {
      if (suppressScrollspy) return;

      if (isAtPageBottom()) {
        setActive(sections[sections.length - 1].link);
        return;
      }

      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

      if (!visible) return;

      setActive(findLinkForSection(visible.target));
    },
    {
      threshold: [0.16, 0.32, 0.48, 0.64],
      rootMargin: "-18% 0px -58% 0px",
    }
  );

  sections.forEach(({ section }) => sectionObserver.observe(section));

  window.addEventListener(
    "hashchange",
    () => {
      const active = links.find((link) => link.hash === window.location.hash);
      if (active) setActive(active, { jump: true });
    },
    { passive: true }
  );

  window.addEventListener("scroll", updateActiveFromScrollPosition, { passive: true });

  window.addEventListener(
    "resize",
    () => {
      const active = sideToc.querySelector("a.is-active");
      if (active) setActive(active, { jump: true });
    },
    { passive: true }
  );
})();
