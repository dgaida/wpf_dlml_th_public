---
layout: page
title: Vorlesungen und Übungen
description: Übersicht über Vorlesungstermine und Übungen des Kurses.
---

# Vorlesungen und Übungen

{% for module in site.modules %}
{{ module }}
{% endfor %}

<script>
document.addEventListener('DOMContentLoaded', function() {
  const links = document.querySelectorAll('a[href$=".ipynb"]');
  const basePrefix = 'https://colab.research.google.com/github/dgaida/wpf_dlml_th_public/blob/main/';
  const localBase = '/wpf_dlml_th_public/';

  links.forEach(link => {
    let href = link.getAttribute('href');
    if (href.startsWith(localBase)) {
      link.href = basePrefix + href.substring(localBase.length);
      link.target = '_blank';
    } else if (href.startsWith('/') && !href.startsWith(localBase) && !href.startsWith('http')) {
        // Handle root-relative links if they exist and are not already prefixed with baseurl
        // Though in this setup, baseurl is usually included in links
    }
  });
});
</script>
