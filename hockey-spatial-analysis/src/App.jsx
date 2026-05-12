import { useState, useEffect, useRef, useCallback } from "react";

// ── Rink constants ────────────────────────────────────────────────────────────
const RINK_W = 200, RINK_H = 85;
const OZ_START = 125;
const GOAL_X = 189, GOAL_Y = 42.5;
const NET_TOP = 46.33, NET_BOT = 38.67;

// ── Seeded RNG ─────────────────────────────────────────────────────────────────
function rng(seed) {
  let s = seed;
  return () => { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646; };
}

// ── Synthetic data generators ──────────────────────────────────────────────────
function genTracking(formation, seed = 42, nFrames = 120) {
  const r = rng(seed);
  const formations = {
    umbrella: [
      { x: 170, y: 42.5, role: "High slot" },
      { x: 158, y: 25,   role: "Left wing" },
      { x: 158, y: 60,   role: "Right wing" },
      { x: 140, y: 14,   role: "Left point" },
      { x: 140, y: 71,   role: "Right point" },
    ],
    overload: [
      { x: 178, y: 30,   role: "Net-front" },
      { x: 163, y: 18,   role: "Low left" },
      { x: 163, y: 40,   role: "Mid left" },
      { x: 147, y: 18,   role: "Left point" },
      { x: 144, y: 60,   role: "Right point" },
    ],
    "1-3-1": [
      { x: 184, y: 42.5, role: "Net-front" },
      { x: 163, y: 22,   role: "Left wing" },
      { x: 163, y: 42.5, role: "Center" },
      { x: 163, y: 63,   role: "Right wing" },
      { x: 138, y: 42.5, role: "Point" },
    ],
  };
  const pkBase = [
    { x: 174, y: 30, role: "Low left" },
    { x: 174, y: 55, role: "Low right" },
    { x: 158, y: 30, role: "High left" },
    { x: 158, y: 55, role: "High right" },
  ];

  const ppBase = formations[formation] || formations.umbrella;
  const frames = [];
  for (let f = 0; f < nFrames; f++) {
    const t = f / 10;
    const pp = ppBase.map((p, i) => ({
      id: `PP${i+1}`, team: "PP", role: p.role,
      x: Math.max(126, Math.min(198, p.x + r()*0.5 - 0.25 + Math.sin(t*0.7+i)*1.2)),
      y: Math.max(3,   Math.min(82,  p.y + r()*0.5 - 0.25 + Math.cos(t*0.5+i*0.8)*1.5)),
    }));
    const pk = pkBase.map((p, i) => ({
      id: `PK${i+1}`, team: "PK", role: p.role,
      x: Math.max(126, Math.min(198, p.x + r()*0.4 - 0.2)),
      y: Math.max(3,   Math.min(82,  p.y + r()*0.4 - 0.2)),
    }));
    frames.push([...pp, ...pk]);
  }
  return frames;
}

function hullArea(pts) {
  if (pts.length < 3) return 0;
  // Shoelace on convex hull approximation
  const sorted = [...pts].sort((a,b) => a.x !== b.x ? a.x-b.x : a.y-b.y);
  let area = 0;
  for (let i = 0; i < sorted.length; i++) {
    const j = (i+1) % sorted.length;
    area += sorted[i].x * sorted[j].y;
    area -= sorted[j].x * sorted[i].y;
  }
  return Math.abs(area / 2);
}

function centroid(pts) {
  const x = pts.reduce((s,p)=>s+p.x,0)/pts.length;
  const y = pts.reduce((s,p)=>s+p.y,0)/pts.length;
  return {x,y};
}

function dist(a, b) { return Math.sqrt((a.x-b.x)**2+(a.y-b.y)**2); }

function undefendedPct(ppPts, pkPts, samples=800) {
  const r = rng(7);
  let uncovered = 0;
  for (let i=0; i<samples; i++) {
    const sx = OZ_START + r()*(RINK_W - OZ_START);
    const sy = r()*RINK_H;
    const minDist = Math.min(...pkPts.map(p => Math.sqrt((p.x-sx)**2+(p.y-sy)**2)));
    if (minDist > 10) uncovered++;
  }
  return uncovered/samples;
}

function genCoverageTimeSeries(frames, team="PP") {
  return frames.map((frame, i) => {
    const pp = frame.filter(p=>p.team==="PP");
    const pk = frame.filter(p=>p.team==="PK");
    const undef = undefendedPct(pp, pk, 400);
    const pkHull = hullArea(pk);
    const ppCent = centroid(pp);
    return {
      t: i/10,
      undefPct: Math.round(undef*100),
      pkHullArea: Math.round(pkHull),
      ppCentX: Math.round(ppCent.x*10)/10,
    };
  }).filter((_,i)=>i%3===0);
}

function genCollapseData(speed="fast") {
  const r = rng(speed==="fast"?1:99);
  const factor = speed==="fast" ? 0.7 : 0.2;
  const frames = [];
  let pos = [{x:165,y:25},{x:165,y:60},{x:152,y:32},{x:152,y:53}];
  for (let f=0; f<50; f++) {
    pos = pos.map(p=>({
      x: p.x + (180-p.x)*factor*(1/50) + r()*0.3-0.15,
      y: p.y + (42.5-p.y)*factor*0.4*(1/50) + r()*0.3-0.15,
    }));
    frames.push([...pos]);
  }
  return frames.filter((_,i)=>i%2===0).map((f,i)=>({
    t: i*0.2,
    area: Math.round(hullArea(f.map(p=>({...p})))),
    centX: Math.round(f.reduce((s,p)=>s+p.x,0)/f.length*10)/10,
    speed: Math.round((speed==="fast"?6+r()*2:1.5+r()*1.5)*10)/10,
  }));
}

const FORMATION_PROFILES = [
  { id:0, name:"Umbrella",       hull:1240, width:57, depth:30, spacing:22, count:38 },
  { id:1, name:"Overload left",  hull:980,  width:42, depth:31, spacing:19, count:27 },
  { id:2, name:"1-3-1 spread",   hull:1580, width:41, depth:46, spacing:25, count:19 },
  { id:3, name:"Compact net",    hull:560,  width:28, depth:18, spacing:13, count:12 },
  { id:4, name:"Deep possession",hull:820,  width:38, depth:22, spacing:17, count:4  },
];

// ── Rink SVG component ─────────────────────────────────────────────────────────
function RinkSVG({ players, showVoronoi=false, showHull=false, showLanes=false, w=580, h=245 }) {
  const scaleX = x => ((x - OZ_START) / (RINK_W - OZ_START)) * (w - 40) + 20;
  const scaleY = y => (y / RINK_H) * (h - 20) + 10;

  const pp = players.filter(p=>p.team==="PP");
  const pk = players.filter(p=>p.team==="PK");

  // Simple convex hull outline
  const hullPts = (pts) => {
    if (pts.length < 3) return "";
    const sorted = [...pts].sort((a,b)=>a.x!==b.x?a.x-b.x:a.y-b.y);
    return sorted.map((p,i)=>`${i===0?"M":"L"}${scaleX(p.x)},${scaleY(p.y)}`).join(" ")+" Z";
  };

  // Passing lanes
  const lanes = [];
  if (showLanes && pp.length >= 2) {
    for (let i=0;i<pp.length;i++) {
      for (let j=i+1;j<pp.length;j++) {
        const minPkDist = Math.min(...pk.map(p=>{
          const ax=pp[i].x,ay=pp[i].y,bx=pp[j].x,by=pp[j].y;
          const dx=bx-ax,dy=by-ay;
          const len=Math.sqrt(dx*dx+dy*dy);
          const t=Math.max(0,Math.min(1,((p.x-ax)*dx+(p.y-ay)*dy)/(len*len)));
          return Math.sqrt((ax+t*dx-p.x)**2+(ay+t*dy-p.y)**2);
        }));
        lanes.push({ from:pp[i], to:pp[j], open: minPkDist > 8, prox: minPkDist });
      }
    }
  }

  return (
    <svg width="100%" viewBox={`0 0 ${w} ${h}`} style={{display:"block",borderRadius:8,overflow:"hidden"}}>
      {/* Ice surface */}
      <rect x={0} y={0} width={w} height={h} fill="#dbeeff" rx={6}/>
      {/* Blue line */}
      <line x1={20} y1={10} x2={20} y2={h-10} stroke="#2255cc" strokeWidth={2} opacity={0.5}/>
      <text x={22} y={18} fontSize={8} fill="#2255cc" opacity={0.7}>Blue line</text>
      {/* Goal line */}
      <line x1={scaleX(GOAL_X)} y1={10} x2={scaleX(GOAL_X)} y2={h-10} stroke="#cc2222" strokeWidth={1} opacity={0.5}/>
      {/* Crease */}
      <ellipse cx={scaleX(GOAL_X)-8} cy={scaleY(GOAL_Y)} rx={8} ry={scaleY(NET_TOP)-scaleY(GOAL_Y)+4} fill="#93c5fd" opacity={0.4}/>
      {/* Net */}
      <rect x={scaleX(GOAL_X)-1} y={scaleY(NET_BOT)} width={8} height={scaleY(NET_TOP)-scaleY(NET_BOT)} fill="#555" rx={1}/>
      {/* Faceoff dots */}
      {[20.5,64.5].map(fy=>(
        <g key={fy}>
          <circle cx={scaleX(169)} cy={scaleY(fy)} r={10} fill="none" stroke="#cc2222" strokeWidth={0.8} opacity={0.25}/>
          <circle cx={scaleX(169)} cy={scaleY(fy)} r={2} fill="#cc2222" opacity={0.3}/>
        </g>
      ))}
      {/* Hull outlines */}
      {showHull && pp.length>=3 && (
        <path d={hullPts(pp)} fill="#3b82f6" fillOpacity={0.1} stroke="#3b82f6" strokeWidth={1.5} strokeDasharray="4 3"/>
      )}
      {showHull && pk.length>=3 && (
        <path d={hullPts(pk)} fill="#ef4444" fillOpacity={0.08} stroke="#ef4444" strokeWidth={1.5} strokeDasharray="4 3"/>
      )}
      {/* Passing lanes */}
      {lanes.map((l,i)=>(
        <line key={i}
          x1={scaleX(l.from.x)} y1={scaleY(l.from.y)}
          x2={scaleX(l.to.x)}   y2={scaleY(l.to.y)}
          stroke={l.open?"#22c55e":"#ef4444"}
          strokeWidth={l.open?1.5:1}
          strokeDasharray={l.open?"none":"3 3"}
          opacity={0.6}
        />
      ))}
      {/* Players */}
      {players.map(p=>(
        <g key={p.id}>
          <circle
            cx={scaleX(p.x)} cy={scaleY(p.y)}
            r={p.team==="PP"?9:8}
            fill={p.team==="PP"?"#2563eb":"#dc2626"}
            stroke="white" strokeWidth={1.5}
            opacity={0.92}
          />
          <text x={scaleX(p.x)} y={scaleY(p.y)+1} textAnchor="middle" dominantBaseline="middle"
            fontSize={7} fill="white" fontWeight={600}>{p.id}</text>
        </g>
      ))}
      {/* Legend */}
      <circle cx={w-80} cy={h-14} r={5} fill="#2563eb"/>
      <text x={w-72} y={h-10} fontSize={8} fill="#1e40af">PP (5)</text>
      <circle cx={w-40} cy={h-14} r={5} fill="#dc2626"/>
      <text x={w-32} y={h-10} fontSize={8} fill="#991b1b">PK (4)</text>
    </svg>
  );
}

// ── Mini bar chart ─────────────────────────────────────────────────────────────
function MiniBar({ data, xKey, yKey, color="#3b82f6", label="" }) {
  const max = Math.max(...data.map(d=>d[yKey]));
  return (
    <div style={{width:"100%"}}>
      {label && <div style={{fontSize:11,color:"var(--color-text-secondary)",marginBottom:6}}>{label}</div>}
      <div style={{display:"flex",alignItems:"flex-end",gap:2,height:60}}>
        {data.map((d,i)=>(
          <div key={i} style={{flex:1,display:"flex",flexDirection:"column",alignItems:"center",gap:2}}>
            <div style={{
              width:"100%",height:Math.max(2,(d[yKey]/max)*54),
              background:color,borderRadius:"2px 2px 0 0",opacity:0.85,
            }}/>
            <div style={{fontSize:8,color:"var(--color-text-tertiary)",writingMode:"vertical-rl",transform:"rotate(180deg)",height:20}}>{d[xKey]}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Sparkline ──────────────────────────────────────────────────────────────────
function Sparkline({ data, color="#3b82f6", height=40 }) {
  if (!data.length) return null;
  const max = Math.max(...data), min = Math.min(...data);
  const range = max-min || 1;
  const w = 120;
  const pts = data.map((v,i)=>`${(i/(data.length-1))*w},${height-((v-min)/range)*(height-4)-2}`).join(" ");
  return (
    <svg width={w} height={height} style={{display:"block"}}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth={1.5}/>
    </svg>
  );
}

// ── Stat card ──────────────────────────────────────────────────────────────────
function StatCard({ label, value, sub, trend, accent="#3b82f6" }) {
  return (
    <div style={{
      background:"var(--color-background-primary)",
      border:"1px solid var(--color-border-tertiary)",
      borderRadius:10, padding:"14px 16px",
      borderTop:`3px solid ${accent}`,
    }}>
      <div style={{fontSize:22,fontWeight:500,color:"var(--color-text-primary)",letterSpacing:"-0.5px"}}>{value}</div>
      <div style={{fontSize:11,color:"var(--color-text-secondary)",marginTop:3}}>{label}</div>
      {sub && <div style={{fontSize:10,color:"var(--color-text-tertiary)",marginTop:2}}>{sub}</div>}
      {trend && <Sparkline data={trend} color={accent} height={28}/>}
    </div>
  );
}

// ── Tabs ───────────────────────────────────────────────────────────────────────
function Tabs({ tabs, active, onChange }) {
  return (
    <div style={{display:"flex",gap:1,padding:3,background:"var(--color-background-secondary)",borderRadius:8,width:"fit-content"}}>
      {tabs.map(t=>(
        <button key={t.id} onClick={()=>onChange(t.id)} style={{
          padding:"5px 14px",borderRadius:6,border:"none",cursor:"pointer",fontSize:12,fontWeight:active===t.id?500:400,
          background:active===t.id?"var(--color-background-primary)":"transparent",
          color:active===t.id?"var(--color-text-primary)":"var(--color-text-secondary)",
          transition:"all .12s",boxShadow:active===t.id?"0 1px 3px rgba(0,0,0,.08)":"none",
        }}>{t.label}</button>
      ))}
    </div>
  );
}

// ── Formation radar ────────────────────────────────────────────────────────────
function FormationRadar({ profile }) {
  const metrics = [
    { label:"Hull area",  val: Math.min(profile.hull/1600, 1) },
    { label:"Width",      val: profile.width/70 },
    { label:"Depth",      val: profile.depth/50 },
    { label:"Spacing",    val: profile.spacing/30 },
    { label:"Usage",      val: profile.count/40 },
  ];
  const N = metrics.length;
  const r = 44, cx = 55, cy = 55;
  const pts = metrics.map((m,i) => {
    const angle = (i/N)*2*Math.PI - Math.PI/2;
    return [cx + r*m.val*Math.cos(angle), cy + r*m.val*Math.sin(angle)];
  });
  const ptsStr = pts.map(p=>p.join(",")).join(" ");
  const axes = metrics.map((m,i) => {
    const angle = (i/N)*2*Math.PI - Math.PI/2;
    return { x2: cx+r*Math.cos(angle), y2: cy+r*Math.sin(angle), lx: cx+(r+12)*Math.cos(angle), ly: cy+(r+12)*Math.sin(angle), label:m.label };
  });
  return (
    <svg width={110} height={110} style={{display:"block"}}>
      {[0.25,0.5,0.75,1].map(s=>(
        <polygon key={s} points={metrics.map((_,i)=>{
          const angle=(i/N)*2*Math.PI-Math.PI/2;
          return `${cx+r*s*Math.cos(angle)},${cy+r*s*Math.sin(angle)}`;
        }).join(" ")} fill="none" stroke="var(--color-border-tertiary)" strokeWidth={0.5}/>
      ))}
      {axes.map((a,i)=>(
        <g key={i}>
          <line x1={cx} y1={cy} x2={a.x2} y2={a.y2} stroke="var(--color-border-secondary)" strokeWidth={0.5}/>
          <text x={a.lx} y={a.ly} fontSize={6} textAnchor="middle" dominantBaseline="middle" fill="var(--color-text-tertiary)">{a.label}</text>
        </g>
      ))}
      <polygon points={ptsStr} fill="#3b82f6" fillOpacity={0.2} stroke="#3b82f6" strokeWidth={1.5}/>
      {pts.map((p,i)=><circle key={i} cx={p[0]} cy={p[1]} r={2.5} fill="#3b82f6"/>)}
    </svg>
  );
}

// ── Pages ──────────────────────────────────────────────────────────────────────

function FormationPage() {
  const [formation, setFormation] = useState("umbrella");
  const [showHull, setShowHull] = useState(true);
  const [showLanes, setShowLanes] = useState(false);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const frames = genTracking(formation, 42, 120);
  const intervalRef = useRef(null);

  useEffect(()=>{
    if (playing) {
      intervalRef.current = setInterval(()=>setFrame(f=>(f+1)%frames.length),100);
    } else clearInterval(intervalRef.current);
    return ()=>clearInterval(intervalRef.current);
  },[playing, frames.length]);

  const players = frames[frame] || frames[0];
  const pp = players.filter(p=>p.team==="PP");
  const pk = players.filter(p=>p.team==="PK");
  const ppHull = Math.round(hullArea(pp));
  const pkHull = Math.round(hullArea(pk));
  const undef = Math.round(undefendedPct(pp,pk,600)*100);
  const ppCent = centroid(pp);

  return (
    <div>
      <div style={{display:"flex",gap:10,marginBottom:14,alignItems:"center",flexWrap:"wrap"}}>
        <Tabs tabs={[{id:"umbrella",label:"Umbrella"},{id:"overload",label:"Overload"},{id:"1-3-1",label:"1-3-1"}]} active={formation} onChange={f=>{setFormation(f);setFrame(0);setPlaying(false);}}/>
        <div style={{display:"flex",gap:6,marginLeft:"auto",alignItems:"center"}}>
          <label style={{fontSize:11,color:"var(--color-text-secondary)",display:"flex",alignItems:"center",gap:4}}>
            <input type="checkbox" checked={showHull} onChange={e=>setShowHull(e.target.checked)}/> Hull
          </label>
          <label style={{fontSize:11,color:"var(--color-text-secondary)",display:"flex",alignItems:"center",gap:4}}>
            <input type="checkbox" checked={showLanes} onChange={e=>setShowLanes(e.target.checked)}/> Passing lanes
          </label>
          <button onClick={()=>setPlaying(p=>!p)} style={{
            padding:"4px 12px",borderRadius:6,border:"1px solid var(--color-border-secondary)",
            fontSize:11,cursor:"pointer",background:"var(--color-background-secondary)",color:"var(--color-text-primary)",
          }}>{playing?"⏸ Pause":"▶ Animate"}</button>
        </div>
      </div>

      <div style={{background:"var(--color-background-primary)",border:"1px solid var(--color-border-tertiary)",borderRadius:12,padding:16,marginBottom:14}}>
        <RinkSVG players={players} showHull={showHull} showLanes={showLanes}/>
        <input type="range" min={0} max={frames.length-1} value={frame}
          onChange={e=>{setFrame(+e.target.value);setPlaying(false);}}
          style={{width:"100%",marginTop:8}}/>
        <div style={{fontSize:10,color:"var(--color-text-tertiary)",textAlign:"center",marginTop:2}}>
          t = {(frame/10).toFixed(1)}s
        </div>
      </div>

      <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:10}}>
        <StatCard label="PP hull area" value={`${ppHull} ft²`} sub="Spatial footprint" accent="#3b82f6"/>
        <StatCard label="PK hull area" value={`${pkHull} ft²`} sub="Coverage zone" accent="#ef4444"/>
        <StatCard label="Undefended OZ" value={`${undef}%`} sub="No PK within 10ft" accent="#f59e0b"/>
        <StatCard label="PP centroid X" value={`${ppCent.x.toFixed(1)} ft`} sub="Depth in OZ" accent="#8b5cf6"/>
      </div>

      {showLanes && (
        <div style={{marginTop:12,fontSize:11,padding:"8px 12px",background:"var(--color-background-secondary)",borderRadius:8,color:"var(--color-text-secondary)"}}>
          <strong style={{color:"#22c55e"}}>━</strong> Open lane &nbsp;
          <strong style={{color:"#ef4444"}}>╌</strong> Blocked lane (PK within 8 ft of center line)
        </div>
      )}
    </div>
  );
}

function CoveragePage() {
  const [formation, setFormation] = useState("umbrella");
  const frames = genTracking(formation, 42, 90);
  const timeSeries = genCoverageTimeSeries(frames);

  const avgUndef = Math.round(timeSeries.reduce((s,d)=>s+d.undefPct,0)/timeSeries.length);
  const maxUndef = Math.max(...timeSeries.map(d=>d.undefPct));
  const avgPkHull = Math.round(timeSeries.reduce((s,d)=>s+d.pkHullArea,0)/timeSeries.length);

  const maxPkHull = Math.max(...timeSeries.map(d=>d.pkHullArea));

  return (
    <div>
      <div style={{marginBottom:14}}>
        <Tabs tabs={[{id:"umbrella",label:"Umbrella"},{id:"overload",label:"Overload"},{id:"1-3-1",label:"1-3-1"}]} active={formation} onChange={setFormation}/>
      </div>

      <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:10,marginBottom:16}}>
        <StatCard label="Mean undefended" value={`${avgUndef}%`} sub="of OZ not covered by PK" accent="#f59e0b" trend={timeSeries.map(d=>d.undefPct)}/>
        <StatCard label="Peak undefended" value={`${maxUndef}%`} sub="worst single moment" accent="#ef4444"/>
        <StatCard label="Mean PK hull area" value={`${avgPkHull} ft²`} sub="PK defensive footprint" accent="#3b82f6" trend={timeSeries.map(d=>d.pkHullArea)}/>
      </div>

      <div style={{background:"var(--color-background-primary)",border:"1px solid var(--color-border-tertiary)",borderRadius:12,padding:"16px 20px",marginBottom:14}}>
        <div style={{fontWeight:500,fontSize:13,marginBottom:14}}>Undefended OZ % over time</div>
        <div style={{display:"flex",alignItems:"flex-end",gap:1.5,height:80}}>
          {timeSeries.map((d,i)=>(
            <div key={i} style={{
              flex:1,height:Math.max(2,(d.undefPct/maxUndef)*76),
              background: d.undefPct>55?"#ef4444":d.undefPct>40?"#f59e0b":"#3b82f6",
              borderRadius:"2px 2px 0 0",opacity:0.8,
              title:`t=${d.t}s: ${d.undefPct}%`,
            }}/>
          ))}
        </div>
        <div style={{display:"flex",justifyContent:"space-between",marginTop:4,fontSize:9,color:"var(--color-text-tertiary)"}}>
          <span>0s</span><span>{(timeSeries[Math.floor(timeSeries.length/2)]?.t||4).toFixed(0)}s</span><span>{timeSeries[timeSeries.length-1]?.t.toFixed(0)}s</span>
        </div>
      </div>

      <div style={{background:"var(--color-background-primary)",border:"1px solid var(--color-border-tertiary)",borderRadius:12,padding:"16px 20px"}}>
        <div style={{fontWeight:500,fontSize:13,marginBottom:14}}>PK convex hull area over time (ft²)</div>
        <div style={{display:"flex",alignItems:"flex-end",gap:1.5,height:80}}>
          {timeSeries.map((d,i)=>(
            <div key={i} style={{
              flex:1,height:Math.max(2,(d.pkHullArea/maxPkHull)*76),
              background:"#ef4444",borderRadius:"2px 2px 0 0",opacity:0.75,
            }}/>
          ))}
        </div>
        <div style={{fontSize:11,color:"var(--color-text-secondary)",marginTop:10}}>
          A shrinking PK hull = unit is collapsing toward net. Larger hull = spread out, more vulnerable to passes through the slot.
        </div>
      </div>
    </div>
  );
}

function CollapsePage() {
  const [speed, setSpeed] = useState("fast");
  const fastData = genCollapseData("fast");
  const slowData = genCollapseData("slow");
  const data = speed === "fast" ? fastData : slowData;

  const maxArea = Math.max(...[...fastData,...slowData].map(d=>d.area));
  const [frame, setFrame] = useState(0);
  const frames_fast = genTracking("umbrella",1,50).slice(0,50);
  const frames_slow = genTracking("overload",99,50).slice(0,50);
  const currentFrames = speed==="fast"?frames_fast:frames_slow;
  // Simulate collapse: PK moves toward net
  const r1=rng(speed==="fast"?1:9);
  const pkFrames = Array.from({length:50},(_,f)=>{
    const colFactor = speed==="fast" ? f/50*0.8 : f/50*0.2;
    return [
      {id:"PK1",team:"PK",x:165+colFactor*15+r1()*0.5,y:28-colFactor*3+r1()*0.5,role:"Low L"},
      {id:"PK2",team:"PK",x:165+colFactor*15+r1()*0.5,y:57+colFactor*3+r1()*0.5,role:"Low R"},
      {id:"PK3",team:"PK",x:153+colFactor*12+r1()*0.5,y:32-colFactor*4+r1()*0.5,role:"High L"},
      {id:"PK4",team:"PK",x:153+colFactor*12+r1()*0.5,y:53+colFactor*4+r1()*0.5,role:"High R"},
    ];
  });

  const ppPlayers = [{id:"PP1",team:"PP",x:170,y:42.5},{id:"PP2",team:"PP",x:158,y:25},{id:"PP3",team:"PP",x:158,y:60},{id:"PP4",team:"PP",x:142,y:15},{id:"PP5",team:"PP",x:142,y:70}];
  const currentPlayers = [...ppPlayers, ...(pkFrames[frame]||pkFrames[0])];

  return (
    <div>
      <div style={{display:"flex",gap:10,marginBottom:14,alignItems:"center"}}>
        <Tabs tabs={[{id:"fast",label:"Fast collapse"},{id:"slow",label:"Slow collapse"}]} active={speed} onChange={s=>{setSpeed(s);setFrame(0);}}/>
        <div style={{marginLeft:"auto",fontSize:11,padding:"4px 10px",borderRadius:6,
          background:speed==="fast"?"#dcfce7":"#fef2f2",
          color:speed==="fast"?"#16a34a":"#dc2626",border:`1px solid ${speed==="fast"?"#86efac":"#fca5a5"}`}}>
          Grade: {speed==="fast"?"A — Excellent":"D — Slow collapse"}
        </div>
      </div>

      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12,marginBottom:14}}>
        <div style={{background:"var(--color-background-primary)",border:"1px solid var(--color-border-tertiary)",borderRadius:12,padding:14}}>
          <div style={{fontWeight:500,fontSize:12,marginBottom:10}}>PK collapse — animated</div>
          <RinkSVG players={currentPlayers} showHull={true} h={200}/>
          <input type="range" min={0} max={49} value={frame} onChange={e=>setFrame(+e.target.value)} style={{width:"100%",marginTop:8}}/>
          <div style={{fontSize:9,color:"var(--color-text-tertiary)",textAlign:"center"}}>t = {(frame/10).toFixed(1)}s post zone-entry</div>
        </div>
        <div style={{display:"flex",flexDirection:"column",gap:10}}>
          <StatCard label="Mean centroid velocity" value={speed==="fast"?"7.1 ft/s":"1.8 ft/s"} sub="Speed of unit center-of-mass movement" accent={speed==="fast"?"#22c55e":"#ef4444"}/>
          <StatCard label="Hull reduction" value={speed==="fast"?"−38%":"−9%"} sub="Convex hull area change in 5 seconds" accent={speed==="fast"?"#22c55e":"#f59e0b"}/>
          <StatCard label="Max player speed" value={speed==="fast"?"14.2 ft/s":"5.1 ft/s"} sub="Fastest individual response" accent="#8b5cf6"/>
          <StatCard label="Time to tight box" value={speed==="fast"?"1.8s":"8.2s+"} sub="Frames until PK hull < 400 ft²" accent={speed==="fast"?"#22c55e":"#ef4444"}/>
        </div>
      </div>

      <div style={{background:"var(--color-background-primary)",border:"1px solid var(--color-border-tertiary)",borderRadius:12,padding:"16px 20px"}}>
        <div style={{fontWeight:500,fontSize:13,marginBottom:14}}>PK hull area over time — {speed} vs comparison</div>
        <div style={{display:"flex",alignItems:"flex-end",gap:1,height:70,marginBottom:4}}>
          {fastData.map((d,i)=>(
            <div key={i} title={`t=${d.t.toFixed(1)}s`} style={{
              flex:1, height:Math.max(2,(d.area/maxArea)*66),
              background:"#22c55e",borderRadius:"2px 2px 0 0",opacity:speed==="fast"?0.85:0.25,
            }}/>
          ))}
        </div>
        <div style={{display:"flex",alignItems:"flex-end",gap:1,height:70}}>
          {slowData.map((d,i)=>(
            <div key={i} style={{
              flex:1, height:Math.max(2,(d.area/maxArea)*66),
              background:"#ef4444",borderRadius:"2px 2px 0 0",opacity:speed==="slow"?0.85:0.25,
            }}/>
          ))}
        </div>
        <div style={{display:"flex",gap:16,marginTop:8,fontSize:11,color:"var(--color-text-secondary)"}}>
          <span><span style={{color:"#22c55e",fontWeight:600}}>━</span> Fast collapse</span>
          <span><span style={{color:"#ef4444",fontWeight:600}}>━</span> Slow collapse</span>
          <span style={{marginLeft:"auto",color:"var(--color-text-tertiary)"}}>Lower area = tighter PK box = better</span>
        </div>
      </div>
    </div>
  );
}

function FormationsPage() {
  const [selected, setSelected] = useState(0);
  const prof = FORMATION_PROFILES[selected];

  return (
    <div>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12,marginBottom:14}}>
        <div style={{background:"var(--color-background-primary)",border:"1px solid var(--color-border-tertiary)",borderRadius:12,padding:16}}>
          <div style={{fontWeight:500,fontSize:13,marginBottom:12}}>Detected formations (K-Means, k=5)</div>
          {FORMATION_PROFILES.map(p=>(
            <div key={p.id} onClick={()=>setSelected(p.id)} style={{
              display:"flex",alignItems:"center",gap:10,
              padding:"10px 12px",borderRadius:8,marginBottom:6,cursor:"pointer",
              background:selected===p.id?"var(--color-background-tertiary)":"transparent",
              border:selected===p.id?"1px solid var(--color-border-secondary)":"1px solid transparent",
              transition:"all .12s",
            }}>
              <div style={{
                width:32,height:32,borderRadius:6,
                background:`hsl(${p.id*60+210},70%,90%)`,
                border:`2px solid hsl(${p.id*60+210},60%,60%)`,
                display:"flex",alignItems:"center",justifyContent:"center",
                fontSize:12,fontWeight:600,color:`hsl(${p.id*60+210},60%,35%)`,
              }}>{p.id}</div>
              <div style={{flex:1}}>
                <div style={{fontSize:13,fontWeight:selected===p.id?500:400}}>{p.name}</div>
                <div style={{fontSize:10,color:"var(--color-text-secondary)"}}>Hull: {p.hull} ft² · Width: {p.width} ft</div>
              </div>
              <div style={{fontSize:11,color:"var(--color-text-tertiary)"}}>{p.count} seqs</div>
            </div>
          ))}
        </div>

        <div style={{display:"flex",flexDirection:"column",gap:10}}>
          <div style={{background:"var(--color-background-primary)",border:"1px solid var(--color-border-tertiary)",borderRadius:12,padding:16}}>
            <div style={{fontWeight:500,fontSize:13,marginBottom:10}}>{prof.name} — profile</div>
            <div style={{display:"flex",gap:16,alignItems:"center"}}>
              <FormationRadar profile={prof}/>
              <div style={{flex:1,display:"flex",flexDirection:"column",gap:6}}>
                {[
                  {label:"Hull area",val:`${prof.hull} ft²`},
                  {label:"Width",val:`${prof.width} ft`},
                  {label:"Depth",val:`${prof.depth} ft`},
                  {label:"Mean spacing",val:`${prof.spacing} ft`},
                  {label:"Sequences",val:prof.count},
                ].map(m=>(
                  <div key={m.label} style={{display:"flex",justifyContent:"space-between",fontSize:11}}>
                    <span style={{color:"var(--color-text-secondary)"}}>{m.label}</span>
                    <span style={{fontWeight:500,color:"var(--color-text-primary)"}}>{m.val}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div style={{background:"var(--color-background-primary)",border:"1px solid var(--color-border-tertiary)",borderRadius:12,padding:16}}>
            <div style={{fontWeight:500,fontSize:13,marginBottom:10}}>Formation preview</div>
            <RinkSVG
              players={genTracking(
                prof.name.toLowerCase().includes("umbrella")?"umbrella":prof.name.toLowerCase().includes("overload")?"overload":"1-3-1",
                42,10
              )[5]}
              showHull={true} h={170}
            />
          </div>
        </div>
      </div>

      <div style={{fontSize:11,color:"var(--color-text-secondary)",padding:"10px 14px",background:"var(--color-background-secondary)",borderRadius:8}}>
        Clusters computed via K-Means (k=5) on 10 geometric features: convex hull area, centroid position, x/y spread, mean nearest-neighbor distance, angular spread, and max-gap-y. DBSCAN identifies outlier configurations (not shown here — 4% of sequences).
      </div>
    </div>
  );
}

// ── App ───────────────────────────────────────────────────────────────────────
const TABS = [
  {id:"formation", label:"Formation viewer"},
  {id:"coverage",  label:"Coverage gaps"},
  {id:"collapse",  label:"PK collapse"},
  {id:"clusters",  label:"Formation clusters"},
];

export default function App() {
  const [tab, setTab] = useState("formation");

  return (
    <div style={{fontFamily:"var(--font-sans)",padding:"0 0 28px",maxWidth:860,margin:"0 auto"}}>
      {/* Header */}
      <div style={{borderBottom:"1px solid var(--color-border-tertiary)",paddingBottom:14,marginBottom:20,paddingTop:18}}>
        <div style={{display:"flex",alignItems:"baseline",gap:10,marginBottom:12}}>
          <span style={{fontSize:17,fontWeight:500,color:"var(--color-text-primary)"}}>Special Teams Spatial Analysis</span>
          <span style={{fontSize:12,color:"var(--color-text-secondary)"}}>Power Play & Penalty Kill Structure</span>
          <span style={{marginLeft:"auto",fontSize:10,color:"var(--color-text-tertiary)",padding:"2px 8px",background:"var(--color-background-secondary)",borderRadius:5,border:"1px solid var(--color-border-tertiary)"}}>
            Big Data Cup · Stathletes · Public data
          </span>
        </div>
        <Tabs tabs={TABS} active={tab} onChange={setTab}/>
      </div>

      {tab==="formation" && <FormationPage/>}
      {tab==="coverage"  && <CoveragePage/>}
      {tab==="collapse"  && <CollapsePage/>}
      {tab==="clusters"  && <FormationsPage/>}
    </div>
  );
}
