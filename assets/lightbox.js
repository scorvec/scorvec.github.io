/* lightbox.js — dependency-free responsive image lightbox.
   Auto-attaches to: img[data-zoom], .zoomable img, figure img, main img  (override: window.LB_SELECTOR)
   Skips: images with data-nozoom, or natural width < 500 px.
   Interactions: click = open · wheel = zoom at cursor · drag = pan · double-click = fit/200 %
                 ← → = prev/next image on the page · Esc or backdrop or × = close · pinch on touch. */
(function(){
'use strict';
var SEL=window.LB_SELECTOR||'img[data-zoom], .zoomable img, figure img, main img';
var imgs=[],ov,pic,cap,idx=0,scale=1,tx=0,ty=0,fitScale=1,drag=null,pinch=null;

function collect(){
  imgs=Array.prototype.slice.call(document.querySelectorAll(SEL))
    .filter(function(im,i,a){return a.indexOf(im)===i && !im.hasAttribute('data-nozoom');});
  imgs.forEach(function(im){
    function arm(){ if((im.naturalWidth||0)<500) return;
      im.classList.add('lb-target');
      if(!im._lb){ im._lb=1; im.addEventListener('click',function(e){e.preventDefault();open(imgs.indexOf(im));}); } }
    if(im.complete) arm(); else im.addEventListener('load',arm);
  });
}
function build(){
  ov=document.createElement('div'); ov.className='lb-overlay';
  ov.innerHTML='<button class="lb-x" aria-label="close">&times;</button>'+
    '<button class="lb-nav lb-prev" aria-label="previous">&#8249;</button>'+
    '<button class="lb-nav lb-next" aria-label="next">&#8250;</button>'+
    '<div class="lb-hint">scroll to zoom &#183; drag to pan &#183; double-click to toggle</div>'+
    '<img alt=""><div class="lb-cap"></div>';
  document.body.appendChild(ov);
  pic=ov.querySelector('img'); cap=ov.querySelector('.lb-cap');
  ov.addEventListener('click',function(e){ if(e.target===ov) close(); });
  ov.querySelector('.lb-x').addEventListener('click',close);
  ov.querySelector('.lb-prev').addEventListener('click',function(){nav(-1);});
  ov.querySelector('.lb-next').addEventListener('click',function(){nav(1);});
  ov.addEventListener('wheel',function(e){
    e.preventDefault();
    var k=Math.exp(-e.deltaY*0.0015), ns=Math.min(Math.max(scale*k,fitScale*0.5),12);
    var r=pic.getBoundingClientRect(),cx=e.clientX-(r.left+r.width/2),cy=e.clientY-(r.top+r.height/2);
    tx-=cx*(ns/scale-1); ty-=cy*(ns/scale-1); scale=ns; apply();
  },{passive:false});
  pic.addEventListener('pointerdown',function(e){
    if(pinch) return;
    e.preventDefault(); pic.setPointerCapture(e.pointerId);
    drag={x:e.clientX,y:e.clientY}; pic.classList.add('lb-panning');
  });
  pic.addEventListener('pointermove',function(e){
    if(!drag) return; tx+=e.clientX-drag.x; ty+=e.clientY-drag.y; drag={x:e.clientX,y:e.clientY}; apply();
  });
  ['pointerup','pointercancel'].forEach(function(ev){
    pic.addEventListener(ev,function(){drag=null;pic.classList.remove('lb-panning');});
  });
  pic.addEventListener('dblclick',function(e){
    e.preventDefault();
    if(scale>fitScale*1.05){ scale=fitScale; tx=ty=0; }
    else{ scale=Math.max(2,fitScale*2); }
    apply();
  });
  // basic pinch
  var pts={};
  ov.addEventListener('pointerdown',function(e){ pts[e.pointerId]=e; if(Object.keys(pts).length===2) pinch=dist(pts); });
  ov.addEventListener('pointermove',function(e){
    if(!(e.pointerId in pts)) return; pts[e.pointerId]=e;
    if(pinch){ var d=dist(pts); scale=Math.min(Math.max(scale*d/pinch,fitScale*0.5),12); pinch=d; apply(); }
  });
  ['pointerup','pointercancel'].forEach(function(ev){
    ov.addEventListener(ev,function(e){ delete pts[e.pointerId]; if(Object.keys(pts).length<2) pinch=null; });
  });
  document.addEventListener('keydown',function(e){
    if(!ov.classList.contains('lb-open')) return;
    if(e.key==='Escape') close();
    else if(e.key==='ArrowLeft') nav(-1);
    else if(e.key==='ArrowRight') nav(1);
  });
  addEventListener('resize',function(){ if(ov.classList.contains('lb-open')) fit(); });
}
function dist(p){ var v=Object.values(p); return Math.hypot(v[0].clientX-v[1].clientX,v[0].clientY-v[1].clientY); }
function fit(){
  fitScale=Math.min((innerWidth-40)/pic.naturalWidth,(innerHeight-70)/pic.naturalHeight,1);
  scale=fitScale; tx=ty=0; apply();
}
function apply(){
  pic.style.transform='translate('+tx+'px,'+ty+'px) scale('+scale+')';
  pic.style.width=pic.naturalWidth+'px';
}
function caption(im){
  var f=im.closest('figure'),c=f&&f.querySelector('figcaption');
  return (c&&c.textContent)||im.getAttribute('alt')||'';
}
function open(i){
  if(!ov) build();
  idx=i; var im=imgs[i];
  pic.src=im.currentSrc||im.src;
  cap.textContent=caption(im);
  document.body.style.overflow='hidden';
  ov.style.display='flex'; requestAnimationFrame(function(){ov.classList.add('lb-open');});
  if(pic.complete) fit(); else pic.onload=fit;
}
function nav(d){
  var n=(idx+d+imgs.length)%imgs.length;
  open(n);
}
function close(){
  ov.classList.remove('lb-open'); document.body.style.overflow='';
  setTimeout(function(){ov.style.display='none';},180);
}
if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',collect); else collect();
window.LB_rescan=collect;   // call after injecting images dynamically
})();
