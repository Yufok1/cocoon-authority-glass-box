#!/usr/bin/env python3
"""Cocoon Mind - lean LIVE neural-network feed of a cocoon's illumination output.

Utility, not novelty: minimal cosmetics, low latency. A continuous D3 force sim that
POLLS a data file and joins new nodes/edges in smoothly, so a running cocoon's mind
grows live and "leaves you behind to catch up." Cheap refraction = new nodes flash in.
A couple color presets (keys 1/2/3). Self-contained HTML; data in a side JSON it polls.

Usage:
    python cocoon_graph.py COCOON.py            # emit cocoon_graph.json + cocoon_mind.html
    # then open cocoon_mind.html; point a running trainer at graph_from_agent() to feed it live.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from cocoon_illumination import CocoonIllumination, _load_cocoon


def graph_from_agent(agent, top: int = 140) -> dict:
    """Build the {nodes,links,meta} feed from a (live or loaded) cocoon agent."""
    illum = CocoonIllumination(agent)
    rels = illum.record.relations
    degree: dict[str, int] = {}
    for r in rels:
        for k in ("source", "target"):
            v = str(r.get(k))
            if v and v != "None":
                degree[v] = degree.get(v, 0) + 1
    top_concepts = [c for c, _ in sorted(degree.items(), key=lambda kv: kv[1], reverse=True)[:top]]
    keep = set(top_concepts)
    nodes = [{"id": c, "label": c, "type": "concept", "degree": degree[c]} for c in top_concepts]
    links, seen = [], set()
    for r in rels:
        s, t = str(r.get("source")), str(r.get("target"))
        if s in keep and t in keep and s != t and (s, t) not in seen:
            seen.add((s, t))
            links.append({"source": s, "target": t, "strength": round(float(r.get("strength", 0.5) or 0.5), 3)})
    names = getattr(agent, "organism_names", []) or illum.agent.architecture.get("organism_names", [])
    for nm in names:
        oid = f"org::{nm}"
        nodes.append({"id": oid, "label": str(nm), "type": "organism", "degree": max(degree.values(), default=10)})
        for a in top_concepts[:3]:
            links.append({"source": oid, "target": a, "strength": 0.9})
    return {"nodes": nodes, "links": links,
            "meta": {"level": illum.level, "level_name": illum.status()["level_name"],
                     "organisms": len(names), "concepts": len(top_concepts), "relations": len(links)}}


def write_json(graph: dict, path: str) -> None:
    Path(path).write_text(json.dumps(graph), encoding="utf-8")


_HTML = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Cocoon Mind</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
 html,body{margin:0;height:100%;background:#04060a;color:#9fb;font-family:Consolas,monospace;overflow:hidden}
 #hud{position:fixed;top:6px;left:10px;z-index:10;font-size:11px;opacity:.85}
 #hud b{color:#3ff;font-size:13px;letter-spacing:1px}
 #panel{position:fixed;top:0;right:0;width:280px;height:100%;background:#060b12ee;border-left:1px solid #1a3;
   padding:10px;box-sizing:border-box;overflow:auto;z-index:10;transform:translateX(100%)}
 #panel.open{transform:translateX(0)}
 #panel h2{color:#3ff;font-size:13px;margin:0 0 6px;word-break:break-all}
 .nb{cursor:pointer;padding:2px 5px;margin:1px 0;border:1px solid #1a3a;font-size:11px}
 .nb:hover{background:#1a3a}
 svg{width:100vw;height:100vh;display:block}
 line{stroke:#1f6feb;stroke-opacity:.18}
 circle{stroke:#000;stroke-width:.4}
 text{font-size:8px;fill:#6a8;pointer-events:none}
</style></head><body>
<div id="hud"><b>COCOON MIND</b> <span id="lvl"></span> &middot; <span id="stat"></span><br>
 keys 1/2/3 = palette &middot; click node to drill &middot; scroll zoom &middot; live</div>
<div id="panel"></div><svg></svg>
<script>
const DATA_URL="__DATAURL__", POLL=1200;
const PAL=[d3.interpolateCool, d3.interpolateViridis, t=>d3.interpolateGreys(0.4+0.6*t)];
let pal=0;
const svg=d3.select("svg"), W=innerWidth, Hh=innerHeight, g=svg.append("g");
svg.call(d3.zoom().scaleExtent([0.1,8]).on("zoom",e=>g.attr("transform",e.transform)));
let nodes=[], links=[], byId=new Map(), adj={};
const sim=d3.forceSimulation([])
  .force("link",d3.forceLink([]).id(d=>d.id).distance(d=>42-30*(d.strength||.3)).strength(.18))
  .force("charge",d3.forceManyBody().strength(-55))
  .force("center",d3.forceCenter(W/2,Hh/2))
  .force("collide",d3.forceCollide().radius(d=>rad(d)+2)).alphaDecay(0.02);
let linkSel=g.append("g").selectAll("line"), nodeSel=g.append("g").selectAll("g");
function rad(d){return d.type==="organism"?11:2.5+Math.sqrt(d.degree);}
function col(d){if(d.type==="organism")return"#ff3ec0";return PAL[pal](Math.min(1,d.degree/40));}
function tick(){linkSel.attr("x1",d=>d.source.x).attr("y1",d=>d.source.y).attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);
 nodeSel.attr("transform",d=>`translate(${d.x},${d.y})`);}
sim.on("tick",tick);
function repaint(){nodeSel.select("circle").attr("fill",col);}
function apply(graph){
 const incoming=new Map(graph.nodes.map(n=>[n.id,n]));
 // keep existing positions, add new
 nodes=nodes.filter(n=>incoming.has(n.id));
 const have=new Set(nodes.map(n=>n.id));
 const fresh=[];
 graph.nodes.forEach(n=>{if(have.has(n.id)){Object.assign(byId.get(n.id),{degree:n.degree});}
   else{n.x=W/2+(Math.random()-.5)*80;n.y=Hh/2+(Math.random()-.5)*80;n.isNew=1;nodes.push(n);fresh.push(n);}});
 byId=new Map(nodes.map(n=>[n.id,n]));
 links=graph.links.filter(l=>byId.has(l.source.id||l.source)&&byId.has(l.target.id||l.target))
   .map(l=>({source:l.source.id||l.source,target:l.target.id||l.target,strength:l.strength}));
 adj={};links.forEach(l=>{(adj[l.source]=adj[l.source]||[]).push({id:l.target,s:l.strength});
   (adj[l.target]=adj[l.target]||[]).push({id:l.source,s:l.strength});});
 sim.nodes(nodes); sim.force("link").links(links);
 linkSel=linkSel.data(links,d=>(d.source.id||d.source)+">"+(d.target.id||d.target)).join("line");
 nodeSel=nodeSel.data(nodes,d=>d.id).join(enter=>{const gg=enter.append("g").call(drag());
   gg.append("circle").attr("r",rad).attr("fill",col).on("click",(e,d)=>drill(d));
   gg.append("text").attr("x",d=>rad(d)+2).attr("y",3).text(d=>d.type==="organism"||d.degree>20?d.label:"");return gg;});
 repaint();
 // cheap refraction: new nodes flash white then settle
 fresh.forEach(n=>{const c=nodeSel.filter(d=>d.id===n.id).select("circle");
   c.attr("fill","#fff").attr("r",rad(n)+4).transition().duration(700).attr("fill",col(n)).attr("r",rad(n));n.isNew=0;});
 sim.alpha(0.5).restart();
 d3.select("#lvl").text(graph.meta.level_name||"");
 d3.select("#stat").text(`${graph.meta.organisms} minds / ${nodes.length} nodes / ${links.length} edges`);
}
function drill(d){const nbs=(adj[d.id]||[]).sort((a,b)=>b.s-a.s).slice(0,30);
 const p=d3.select("#panel").classed("open",true);
 p.html(`<h2>${d.label}</h2><div>type ${d.type} · degree ${d.degree}</div><div style="margin-top:8px;opacity:.6">neighbors</div>`);
 nbs.forEach(nb=>p.append("div").attr("class","nb").text(`${nb.id} (${nb.s})`).on("click",()=>{const t=byId.get(nb.id);if(t)drill(t);}));}
function drag(){return d3.drag().on("start",(e,d)=>{if(!e.active)sim.alphaTarget(.2).restart();d.fx=d.x;d.fy=d.y;})
 .on("drag",(e,d)=>{d.fx=e.x;d.fy=e.y;}).on("end",(e,d)=>{if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null;});}
addEventListener("keydown",e=>{if(["1","2","3"].includes(e.key)){pal=+e.key-1;repaint();}});
function poll(){fetch(DATA_URL,{cache:"no-store"}).then(r=>r.json()).then(apply).catch(()=>{});}
poll(); setInterval(poll, POLL);
</script></body></html>"""


def render(cocoon_path: str, html_out="cocoon_mind.html", json_out="cocoon_graph.json", top=140) -> dict:
    mod = _load_cocoon(cocoon_path)
    agent = mod.CocoonAgent()
    graph = graph_from_agent(agent, top=top)
    write_json(graph, json_out)
    Path(html_out).write_text(_HTML.replace("__DATAURL__", Path(json_out).name), encoding="utf-8")
    print(f"[OK] {graph['meta']['level_name']}: {graph['meta']['organisms']} minds, "
          f"{graph['meta']['concepts']} concepts, {graph['meta']['relations']} edges")
    print(f"     feed -> {json_out} | open -> {html_out} (it polls the feed live)")
    return graph["meta"]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); raise SystemExit(2)
    render(sys.argv[1])
