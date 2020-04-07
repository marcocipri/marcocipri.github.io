---
title:  meeting LXC on Ubuntu Core
tags:
  - linux containers
  - linux
  - ubuntu core
  - nuc
---

The real topic of this post should be "_how a rash buy has becomes an exciting surprise_".  
I was quite annoyed making new virtual machines on my laptop and I needed an environment in order to test some Ansible scripts I made.

Searching a solution I find out, on the second hand market, a bare bone solution from Intel : NUC Kit NUC8i5BEH. 
Small less my 2007 old but glorious Mac Mini and cheap enough... 


<!--more-->

## begin :
I need  a small cube in a corner of the house, no screen, neither keyboard but only an ethernet plug. Once the Nuc has been in my hands, comes temporarily connected to the living room TV just for the time needed to install a linux distro…. My idea was to use the NUC as base for a Virtual Box’s host but this plan would been changed soon. I’ve followed the instructions in Ubuntu Core on NUC instead the Ubuntu Desktop on NUC. The process is quite basic then no problem and, at the end of installation,  I pushed my public ssl key on the Canonical's cloud and my NUC is immediatly accessible  to me by ssh. I’ve never faced Ubuntu Core before and the impact was quite odd. At first time Ubuntu Core seems cheap and basic. I’ve spent some time to understand the philosopy of Snap tool as package manager, I'm used to apt on Ubuntu. 
After some investigation was clear that Linux Container LXC/LXD are the natural choise for virtualization on Ubuntu Core. Following this approach I can avoid the creation of a a cluster of virtual machines based on Virtual Box. Lxc allows to create a flock of containers at operating system-level (Linux) more economic than virtual machines.

With LXD I'd create the host and then many containers on it. By default, all containers run in a private network on the host. My needing is to access the containers in the same way as in case of phisical servers and control them whit  ssh term.
On this containers I'd deploy applications by Dockers which run on a LXD host .
Another requirement is to realize some kind of virtual networking in order to control the LXD host and the Linux containers by ssh and access to all the Docker services.  

Here the map of the plan...

<div class="mxgraph" style="max-width:100%;border:1px solid transparent;" data-mxgraph="{&quot;highlight&quot;:&quot;#0000ff&quot;,&quot;nav&quot;:true,&quot;resize&quot;:true,&quot;toolbar&quot;:&quot;zoom layers lightbox&quot;,&quot;edit&quot;:&quot;_blank&quot;,&quot;xml&quot;:&quot;&lt;mxfile host=\&quot;Chrome\&quot; modified=\&quot;2020-04-07T21:17:24.899Z\&quot; agent=\&quot;5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36\&quot; version=\&quot;12.9.9\&quot; etag=\&quot;KDDL_Rwyl_ufvSrVBFDM\&quot; type=\&quot;device\&quot;&gt;&lt;diagram id=\&quot;v351XhcDHZ_iWQRifhIQ\&quot; name=\&quot;Page-1\&quot;&gt;5VpNd5s4FP01XiYHkAF7Gdv1dNGZpseLprOTQQZNAFEhYru/vhJIfFg4dltsJ0w25j0kId373pV4YQTm8e4vCtPwb+KjaGQZ/m4EFiPLMsEY8B/h2UuPA9zSE1DsS1/tWOEfSDoN6c2xj7JWQ0ZIxHDadnokSZDHWj5IKdm2m21I1H5qCgOkOVYejHTvV+yzsPROLLf2f0Q4CNWTTWda3omhaixXkoXQJ9uGC3wYgTklhJVX8W6OIoGewqXstzxyt5oYRQk7p8N0S2a77/liHdifH//5d/c1+JLfyVFeYJTLBcdITpftFQZ85qm4zOPowWOEjsDsBVGGOUqf4BpFjyTDDJOEN1kTxkjMG0Tixgx6zwEleeLPSST68dHApvhrjPEQ4UD0ZSTl3pDFETdMfklyFuEEzStqDe6UE+Z90e4oEmaFL49MRGLE6J43kR1syYiMScuwS3tbEwxkk7DBrSN9UIZUUA1co84vJPC/QMJYQ7zADPlyyYSykAQk4XATgVEBzn+Isb3MF5gz0oYOJf6DiH5uJiRBpWeJxbyKITlSdP8kjHvXVva3wjaVudg1Gy/2TesRUcwXj6h0ZoyS5ypDQOVp8L5cTqfLZcUg8rXEO+CPI0Jy6qFXkJPqwiANEDsV5no8UBRBhl/a8+iiV3Z9JJjPsIojx27HEbDbI5TTkp0OYqSaxe+HjTnRkne1+qjFEs8T1g4OKBPO43gj2pGJMfZ90X1GUYZ/wHUxlKA5FWspVmfPRvbitXSUgiw71zLYpPl4NhzN3TvjHvB9pQW8tH6XTdWEbDYZughTQCNqZDmRYGVDiil4VZo433OxIcwKoTT4X9PlBOL30xNvZ6gR1lT5Q5Ix5eXTLIcub70iMCIgtiFmaJXCIte2fCdvx8uG60ZLwJHjeV0p7rvTtdGTSFdyK0m+U5LcUGnT6ZBpAC6l085QdLpBGSh0uVPB0Q6zJzUYvy4fa0urfqow1EN71Hb7bWn7wQiX1nZ3kNrunNR24wB4dUJ9u+JuHxX3LIWJUmB+kM13JXQM8kMtbSh1s12vSu3baOKPu9J+Yq2B4/Sk1JO2UjsdQm1d8zyt3h0bKApFWklTCm0P2t2vNp/aC844Y1dqfmPtdq+k3X8WJs4gNdY0ukGtRda01faiRNZ66yKrb4cdgvrulfPOurV0TgesnOedentUwcm7UMHpIFVwekoELddqVxHevAbq5Z7/+UHTMs6VS/s4jX+UPOphDU584j33vBkhk+PrdqE7dVwAe0IX2Kc3I7sDXOtSe5FawhDBvTm21nCwPecUdV1w9XrvYMC9Obbj4WB7KAodhZPrYquXsgaDrdlxVrguuHq14d2Ce0bF77rY6uWBwWB7+8DV3zveLbjaMWysf/pxXXD1t+/hgHvryFX/CB4CuNo57NaRCwb0dnZOYeG64OqHBYYohRtCY/3DD5hkWBTItBskRcmKQe9Z4+Sy5TxR8c3K4q/ZD0NVLlcMufd6ArgdHNm/zhE3688yy+pd/XUr+PAT&lt;/diagram&gt;&lt;/mxfile&gt;&quot;}"></div>
<script type="text/javascript" src="https://app.diagrams.net/js/viewer.min.js"></script>

I'd like to be able to create and destroy networks and containers, deploy applications using terraform, ansible and openstack.

<div>
<a href="https://asciinema.org/a/14"><img src="https://asciinema.org/a/14.png" width="400"/></a>
</div>

[![asciicast](https://asciinema.org/a/17648.svg)](https://asciinema.org/a/17648)

<!--more-->
