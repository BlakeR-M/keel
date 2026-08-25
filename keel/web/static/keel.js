/* Keel web, browser side. Four independent pieces, each of which starts only when the page it
   belongs to is on screen, so one missing element never stops the others.

   1. The chat form posts by fetch and swaps the result partial in place, so the page stays put while
      the model works. Without JavaScript the same form posts normally and the server renders the
      whole page with the result in it. Ctrl+Enter (Cmd+Enter on a Mac) sends the form, and a small
      elapsed counter keeps the wait honest on a CPU model.
   2. The permission comparison asks one restricted question as both demo users, side by side,
      through the same /ask path a typed question takes.
   3. The air-gap probe posts a host and renders what each guarded layer did with it.
   4. The redaction demonstration posts text and renders it back with the identifiers replaced.

   Everything rendered from a response is escaped here before it reaches innerHTML, except the /ask
   result partial, which the server rendered through the same escaping template the page itself uses. */
(function () {
  "use strict";

  // ------------------------------------------------------------------ shared helpers

  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  /* Renders the busy state into a container and returns the interval that keeps its elapsed
     counter moving; the caller clears it when the response lands. */
  function startWorking(container, label) {
    var started = Date.now();
    container.innerHTML =
      '<article class="result" aria-busy="true"><p class="working"><span class="pulse" aria-hidden="true"></span>' +
      "<span>" + escapeHtml(label) + "</span>" +
      '<span class="elapsed">0 s</span></p></article>';
    var elapsed = container.querySelector(".elapsed");
    return window.setInterval(function () {
      elapsed.textContent = Math.round((Date.now() - started) / 1000) + " s";
    }, 1000);
  }

  function errorMarkup(message) {
    return '<article class="result error"><header class="result-head"><span class="tag state-error">error</span></header>' +
      '<p class="answer-text">' + escapeHtml(message ||
        "The request stopped before it reached the server. Check that Keel is running, then send it again.") +
      "</p></article>";
  }

  function postAsk(body, onHtml, onError) {
    fetch("/ask", {
      method: "POST",
      body: body,
      headers: { "X-Keel-Partial": "1", "Accept": "text/html" }
    }).then(function (response) {
      return response.text();
    }).then(onHtml).catch(onError);
  }

  function postJson(path, payload) {
    return fetch(path, {
      method: "POST",
      body: JSON.stringify(payload),
      headers: { "Content-Type": "application/json", "Accept": "application/json" }
    }).then(function (response) {
      return response.json().then(function (body) { return { ok: response.ok, body: body }; });
    });
  }

  var modern = window.fetch && window.FormData;

  // ------------------------------------------------------------------ 1. the chat form

  var form = document.getElementById("ask-form");
  if (form && modern) {
    var result = document.getElementById("result");
    var button = form.querySelector("button[type=submit]");
    var question = form.querySelector("textarea[name=question]");
    var timer = null;

    if (question) {
      question.addEventListener("keydown", function (event) {
        if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
          event.preventDefault();
          if (typeof form.requestSubmit === "function") { form.requestSubmit(); } else { button.click(); }
        }
      });
    }

    function stopWorking() {
      if (timer) { window.clearInterval(timer); timer = null; }
      button.disabled = false;
    }

    form.addEventListener("submit", function (event) {
      event.preventDefault();
      var mode = form.querySelector("input[name=mode]:checked");
      var label = mode && mode.value === "agent" ? "Running the agent." : "Working on it.";
      button.disabled = true;
      timer = startWorking(result, label);
      postAsk(new FormData(form), function (html) {
        stopWorking();
        result.innerHTML = html;
        result.scrollIntoView({ behavior: "smooth", block: "start" });
      }, function () {
        stopWorking();
        result.innerHTML = errorMarkup();
      });
    });
  }

  /* The intro dialog shows once per browser and stays reachable from the banner link. The seen flag
     is written at show time, so any way of dismissing it counts. With storage blocked it stays
     closed rather than greeting the visitor on every load. */
  var intro = document.getElementById("intro");
  var introOpen = document.getElementById("intro-open");
  if (intro && typeof intro.showModal === "function") {
    var INTRO_KEY = "keel-demo-intro-seen";
    var seen = true;
    try { seen = window.localStorage.getItem(INTRO_KEY) === "1"; } catch (e) { seen = true; }
    if (!seen) {
      intro.showModal();
      try { window.localStorage.setItem(INTRO_KEY, "1"); } catch (e) { /* storage blocked */ }
    }
    if (introOpen) {
      introOpen.addEventListener("click", function (event) {
        event.preventDefault();
        intro.showModal();
      });
    }
  }

  /* The getting-started panel remembers being folded away, so an operator who knows the answers
     stops being told them. Storage blocked or absent leaves it as the markup found it. */
  var start = document.getElementById("getting-started");
  if (start) {
    var START_KEY = "keel-getting-started-open";
    try {
      var stored = window.localStorage.getItem(START_KEY);
      if (stored !== null) { start.open = stored === "1"; }
    } catch (e) { /* storage blocked */ }
    start.addEventListener("toggle", function () {
      try { window.localStorage.setItem(START_KEY, start.open ? "1" : "0"); } catch (e) { /* storage blocked */ }
    });
  }

  /* Any form carrying data-confirm asks before it submits. Removing a document is the one
     irreversible action in the interface, so it says what it is about to take with it. Without
     JavaScript the form still posts, and the removal still lands in the ledger. */
  document.querySelectorAll("form[data-confirm]").forEach(function (form) {
    form.addEventListener("submit", function (event) {
      if (!window.confirm(form.getAttribute("data-confirm"))) { event.preventDefault(); }
    });
  });

  // ------------------------------------------------------------------ 2. permission comparison

  /* One click asks the restricted question as both demo users. Each column is a real request through
     the same engine, so the refusal and the cited answer land as they run. */
  var compare = document.getElementById("demo-compare");
  var compareSection = document.getElementById("compare");
  if (compare && compareSection && modern) {
    compare.addEventListener("click", function () {
      compare.disabled = true;
      compareSection.innerHTML =
        '<div class="compare-grid">' +
        '<div class="compare-col"><h3 class="compare-head">asked as public</h3><div class="compare-result"></div></div>' +
        '<div class="compare-col"><h3 class="compare-head">asked as hr-officer</h3><div class="compare-result"></div></div>' +
        "</div>";
      var columns = compareSection.querySelectorAll(".compare-result");
      var pending = 2;
      function settle() {
        pending -= 1;
        if (pending === 0) { compare.disabled = false; }
      }
      ["public", "hr-officer"].forEach(function (userId, index) {
        var container = columns[index];
        var tick = startWorking(container, "Asking as " + userId + ".");
        var body = new FormData();
        body.append("question", compare.getAttribute("data-question"));
        body.append("user_id", userId);
        body.append("mode", "answer");
        postAsk(body, function (html) {
          window.clearInterval(tick);
          container.innerHTML = html;
          settle();
        }, function () {
          window.clearInterval(tick);
          container.innerHTML = errorMarkup();
          settle();
        });
      });
      compareSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
  }

  // ------------------------------------------------------------------ 3. the air-gap probe

  var airgapForm = document.getElementById("airgap-form");
  var airgapOut = document.getElementById("airgap-result");
  var airgapHost = document.getElementById("airgap-host");

  function outcomeTag(attempt) {
    if (attempt.refused) { return '<span class="tag state-refused">refused</span>'; }
    if (attempt.outcome === "no-lookup") { return '<span class="tag">no lookup needed</span>'; }
    return '<span class="tag state-error">' + escapeHtml(attempt.outcome || "open") + "</span>";
  }

  function attemptMarkup(attempt) {
    var via = attempt.refused && attempt.via && attempt.via !== attempt.layer
      ? '<span class="layer-via">refused by the ' + escapeHtml(attempt.via) + " guard</span>"
      : "";
    return '<li class="layer' + (attempt.refused ? " is-refused" : "") + '">' +
      '<div class="layer-head"><span class="layer-name mono">' + escapeHtml(attempt.layer) + "</span>" +
      outcomeTag(attempt) + via + "</div>" +
      '<p class="layer-detail mono">' + escapeHtml(attempt.detail) + "</p>" +
      '<p class="layer-note">' + escapeHtml(attempt.note) + "</p>" +
      "</li>";
  }

  function probeMarkup(body) {
    var refusedAll = body.attempts && body.attempts.length &&
      body.attempts.every(function (a) { return a.refused || a.outcome === "no-lookup"; });
    var head = '<article class="result' + (body.allowed ? "" : (refusedAll ? "" : " error")) + '">' +
      '<header class="result-head">' +
      '<span class="tag mode">air-gap ' + (body.guard ? "on" : "off") + "</span>" +
      '<span class="tag mono">' + escapeHtml(body.host) + "</span>" +
      (body.allowed ? '<span class="tag state-approved">on the allow list</span>' : "") +
      "</header>" +
      '<p class="answer-text">' + escapeHtml(body.summary) + "</p>";
    var layers = body.attempts && body.attempts.length
      ? '<ul class="layers">' + body.attempts.map(attemptMarkup).join("") + "</ul>"
      : "";
    var foot = '<p class="result-foot"><span>allow list: <span class="mono">' +
      escapeHtml((body.allow_hosts || []).join(", ")) + "</span></span></p>";
    return head + layers + foot + "</article>";
  }

  function runProbe() {
    var host = (airgapHost.value || "").trim();
    if (!host) { return; }
    var tick = startWorking(airgapOut, "Attempting " + host + " under the guard.");
    postJson("/api/airgap-probe", { host: host }).then(function (response) {
      window.clearInterval(tick);
      airgapOut.innerHTML = response.body && response.body.error
        ? errorMarkup(response.body.error)
        : probeMarkup(response.body);
    }).catch(function () {
      window.clearInterval(tick);
      airgapOut.innerHTML = errorMarkup();
    });
  }

  if (airgapForm && airgapOut && airgapHost && modern) {
    airgapForm.addEventListener("submit", function (event) {
      event.preventDefault();
      runProbe();
    });
    airgapForm.querySelectorAll("button[data-host]").forEach(function (chip) {
      chip.addEventListener("click", function () {
        airgapHost.value = chip.getAttribute("data-host");
        runProbe();
      });
    });
  }

  // ------------------------------------------------------------------ 4. redaction

  var redactForm = document.getElementById("redact-form");
  var redactOut = document.getElementById("redact-result");
  var redactText = document.getElementById("redact-text");

  function redactMarkup(body) {
    var kinds = Object.keys(body.counts || {});
    var tags = kinds.length
      ? kinds.map(function (kind) {
          return '<span class="tag acl mono">' + escapeHtml(kind) + " &times;" + body.counts[kind] + "</span>";
        }).join("")
      : '<span class="tag">nothing matched</span>';
    return '<article class="result">' +
      '<header class="result-head">' + tags + "</header>" +
      '<pre class="code redacted">' + escapeHtml(body.redacted) + "</pre>" +
      '<p class="result-foot"><span>checked for: <span class="mono">' +
      escapeHtml((body.kinds || []).join(", ")) + "</span></span>" +
      "<span>" + (body.findings || []).length + " replaced</span></p>" +
      "</article>";
  }

  if (redactForm && redactOut && redactText && modern) {
    redactForm.addEventListener("submit", function (event) {
      event.preventDefault();
      var tick = startWorking(redactOut, "Redacting.");
      postJson("/api/redact", { text: redactText.value }).then(function (response) {
        window.clearInterval(tick);
        redactOut.innerHTML = response.body && response.body.error
          ? errorMarkup(response.body.error)
          : redactMarkup(response.body);
      }).catch(function () {
        window.clearInterval(tick);
        redactOut.innerHTML = errorMarkup();
      });
    });
  }
})();
