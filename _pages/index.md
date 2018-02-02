---
layout: defaults/page
permalink: index.html
narrow: true
---

{% include components/intro.md %}

[More about Atul.]({{ site.baseurl}}{% link _pages/about.md %})

<!-- <div class="card mb-3">
    <img class="card-img-top" src = "https://drscdn.500px.org/photo/143841823/q%3D80_m%3D1500/v2?webp=true&sig=3c0fa3b75ac4098ca23a31d681a6bc4d2dc91877294fd966dd202aa5b5eb3688"/>
    <div class="card-body bg-light">
        <div class="card-text">A picture from when John was on holiday in the Peak District.</div>
    </div>
</div> -->

### Projects

When I have time betweeen my work and school, I love working on data related projects and learning new skills. [The full list is here]({{ site.baseurl }}{% link list/projects.md %}).

### Posts

This is where I intend to post my thoughts, ideas and updates. There's the three most-recent posts below, or here's [all posts by year.]({{ site.baseurl }}{% link list/posts.html %})

### Recent Posts

{% for post in site.posts limit:3 %}
{% include components/post-card.html %}
{% endfor %}
