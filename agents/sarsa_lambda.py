"""Manual on-policy SARSA(lambda) with replacing eligibility traces."""
from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping
import uuid

import numpy as np

from environments.maze import Action, EventType, MazeEpisode, MazeMDP, MazeSpec, State
from .common import (ACTION_NAMES, ACTION_ORDER, LoadedSarsaLambda, dense_q_array,
                     load_sarsa_lambda_npz, map_checksum, reachable_state_mask,
                     save_sarsa_lambda_npz, state_index, terminal_state_mask,
                     valid_state_mask)
from .q_learning import (ACTION_INDEX, BEHAVIOR_POLICY, Q_INITIALIZATION,
                         SEED_DERIVATION, SHAPING_VERSION, SUPPORTED_SHAPING_METHOD,
                         ActionSelection, EpisodeMetrics, epsilon_for_episode,
                         select_epsilon_greedy)

ALGORITHM_ID = "manual_on_policy_sarsa_lambda"
RUN_CONFIG_SCHEMA_VERSION = CSV_SCHEMA_VERSION = MANIFEST_SCHEMA_VERSION = 1
TRACE_TYPE = "replacing"
TRACE_UPDATE_ORDER = "replace_current_update_active_decay_or_clear"
TRACE_RESET = "clear_at_every_episode_boundary"
TRUNCATION_SEMANTICS = "select_next_action_bootstrap_update_then_end_without_execution"

@dataclass(frozen=True, slots=True)
class SarsaLambdaConfig:
    gamma: float = .95
    alpha: float = .1
    trace_lambda: float = .7
    episodes: int = 5000
    epsilon_start: float = 1.0
    epsilon_end: float = .05
    decay_episodes: int = 4000
    schedule: str = "linear"
    reward_mode: str = "shaped"
    diagnostic_episode: int = 1
    shaping_method: str = SUPPORTED_SHAPING_METHOD
    shaping_version: int = SHAPING_VERSION
    def __post_init__(self):
        if not 0 <= self.gamma < 1: raise ValueError("gamma must be in [0, 1)")
        if not 0 < self.alpha <= 1: raise ValueError("alpha must be in (0, 1]")
        if not 0 <= self.trace_lambda <= 1: raise ValueError("trace_lambda must be in [0, 1]")
        if isinstance(self.episodes, bool) or not isinstance(self.episodes, int) or self.episodes <= 0: raise ValueError("episodes must be a positive integer")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1: raise ValueError("invalid epsilon values")
        if isinstance(self.decay_episodes, bool) or not isinstance(self.decay_episodes, int) or self.decay_episodes < 2: raise ValueError("decay_episodes must be at least 2")
        if self.schedule not in {"linear", "exponential", "geometric"}: raise ValueError("unsupported schedule")
        if self.reward_mode not in {"sparse", "shaped"}: raise ValueError("unsupported reward mode")
        if not 1 <= self.diagnostic_episode <= self.episodes: raise ValueError("diagnostic_episode must be within episodes")
        if self.schedule != "linear" and self.epsilon_end == 0: raise ValueError("exponential epsilon_end must be positive")
        if self.shaping_method != SUPPORTED_SHAPING_METHOD or self.shaping_version != SHAPING_VERSION: raise ValueError("unsupported shaping semantics")

@dataclass(frozen=True, slots=True)
class SarsaLambdaSeeds:
    root: int; behavior: int; transition: int
    def __post_init__(self):
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in (self.root,self.behavior,self.transition)): raise ValueError("seeds must be nonnegative integers")
        if self.behavior == self.transition: raise ValueError("seed streams must be independent")

@dataclass(frozen=True, slots=True)
class SarsaLambdaRunIdentity:
    config_json: str; semantic_config_hash: str; run_id: str
    def __post_init__(self):
        parsed=json.loads(self.config_json); canonical=_canonical_json(parsed)
        digest=hashlib.sha256(canonical.encode()).hexdigest()
        if canonical != self.config_json or digest != self.semantic_config_hash or self.run_id != f"sarsa-lambda-{digest}": raise ValueError("invalid SARSA run identity")
    @property
    def short_id(self): return self.semantic_config_hash[:12]

@dataclass(frozen=True, slots=True)
class TraceUpdate:
    state: State; intended_action: Action; action_index: int; reward: float
    next_state: State; next_action: Action | None; next_action_index: int | None
    old_current_q: float; next_q: float; bootstrap_value: float; target: float
    td_error: float; active_trace_count: int; terminated: bool; truncated: bool

@dataclass(frozen=True, slots=True)
class TraceDiagnosticRow:
    episode: int; step: int; epsilon: float; exploring: bool
    source_key: int; source_row: int; source_col: int
    intended_action: str; intended_action_index: int; actual_action: str; actual_action_index: int
    transition_probability: float; next_key: int; next_row: int; next_col: int
    events: str; base_reward: float; shaping_reward: float; total_reward: float; learning_reward: float
    next_action: str; next_action_index: int; next_exploring: bool
    bootstrap_value: float; target: float; td_error: float
    trace_key: int; trace_row: int; trace_col: int; trace_action: str; trace_action_index: int
    eligibility: float; old_q: float; q_increment: float; new_q: float; post_step_eligibility: float
    terminated: bool; truncated: bool; traces_reset: bool
    def __post_init__(self):
        if self.new_q != self.old_q + self.q_increment: raise ValueError("diagnostic Q reconstruction failed")
        if self.q_increment != self.eligibility * self.td_error * self.alpha: raise ValueError("diagnostic increment reconstruction failed")
        expected=0.0 if self.traces_reset else self.gamma*self.trace_lambda*self.eligibility
        if self.post_step_eligibility != expected: raise ValueError("diagnostic trace decay reconstruction failed")
    alpha: float = .1
    gamma: float = .95
    trace_lambda: float = .7

@dataclass(frozen=True, slots=True)
class SarsaLambdaResult:
    q_values: np.ndarray; state_visit_counts: np.ndarray; state_action_visit_counts: np.ndarray
    valid_state_mask: np.ndarray; reachable_state_mask: np.ndarray; terminal_state_mask: np.ndarray
    episode_metrics: tuple[EpisodeMetrics,...]; diagnostic_rows: tuple[TraceDiagnosticRow,...]
    config: SarsaLambdaConfig; seeds: SarsaLambdaSeeds; identity: SarsaLambdaRunIdentity
    runtime_seconds: float; map_checksum: str
    def __post_init__(self):
        shape=self.valid_state_mask.shape; qshape=(*shape,len(ACTION_ORDER))
        if self.q_values.shape != qshape or self.state_action_visit_counts.shape != qshape or self.state_visit_counts.shape != shape: raise ValueError("invalid SARSA array shapes")
        if self.q_values.dtype != np.float64 or self.state_visit_counts.dtype != np.int64 or self.state_action_visit_counts.dtype != np.int64: raise ValueError("invalid SARSA array dtypes")
        if len(self.episode_metrics) != self.config.episodes: raise ValueError("episode metric count mismatch")
        if any(r.episode != self.config.diagnostic_episode for r in self.diagnostic_rows): raise ValueError("diagnostic episode mismatch")
        if np.any(self.state_action_visit_counts[self.terminal_state_mask]): raise ValueError("terminal actions must not be selected")
    def metadata(self):
        r=json.loads(self.identity.config_json); l=r["learning"]; s=r["seeds"]
        return {"run_id":self.identity.run_id,"semantic_config_hash":self.identity.semantic_config_hash,"run_config_json":self.identity.config_json,
          "student_id":r["student"]["student_id"],"base_seed":r["student"]["base_seed"],"map_checksum":r["map"]["checksum"],"rows":r["map"]["rows"],"cols":r["map"]["cols"],"max_steps":r["map"]["max_steps"],
          "gamma":l["gamma"],"alpha":l["alpha"],"lambda":l["lambda"],"episodes":l["episodes"],"epsilon_start":l["epsilon_start"],"epsilon_end":l["epsilon_end"],"decay_episodes":l["decay_episodes"],"schedule":l["schedule"],"reward_mode":l["reward_mode"],"diagnostic_episode":l["diagnostic_episode"],
          "root_seed":s["root"],"behavior_seed":s["behavior"],"transition_seed":s["transition"],"seed_derivation":s["derivation"],"trace_type":TRACE_TYPE,"trace_update_order":TRACE_UPDATE_ORDER,"trace_reset":TRACE_RESET,"truncation_semantics":TRUNCATION_SEMANTICS,
          "shaping_method":r["shaping"]["method"],"shaping_version":r["shaping"]["version"],"shaping_scale":r["shaping"]["scale"],"behavior_policy":BEHAVIOR_POLICY,"q_initialization":Q_INITIALIZATION,"action_order_json":_canonical_json(r["actions"]["order"]),
          "runtime_seconds":self.runtime_seconds,"total_steps":sum(x.steps for x in self.episode_metrics),"total_successes":sum(x.success for x in self.episode_metrics),"total_terminated":sum(x.terminated for x in self.episode_metrics),"total_truncated":sum(x.truncated for x in self.episode_metrics),"state_visit_total":int(self.state_visit_counts.sum())}

@dataclass(frozen=True, slots=True)
class SarsaLambdaBundlePaths:
    model: Path; episode_metrics: Path; diagnostic: Path; manifest: Path
    def all_paths(self): return self.model,self.episode_metrics,self.diagnostic,self.manifest

def _canonical_json(x): return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False)
def derive_sarsa_lambda_seeds(root):
    if isinstance(root,bool) or not isinstance(root,int) or root<0: raise ValueError("root seed must be nonnegative integer")
    a,b=np.random.SeedSequence(root).spawn(2)
    return SarsaLambdaSeeds(root,int(a.generate_state(1,dtype=np.uint64)[0]),int(b.generate_state(1,dtype=np.uint64)[0]))
def build_sarsa_lambda_run_identity(mdp,config,seeds):
    rw=mdp.rewards
    resolved={"schema_version":1,"algorithm":ALGORITHM_ID,"student":{"student_id":mdp.spec.student_id,"base_seed":mdp.spec.base_seed},"map":{"checksum":map_checksum(mdp.spec),"rows":mdp.spec.rows,"cols":mdp.spec.cols,"max_steps":mdp.spec.max_steps},"transitions":{"intended_probability":mdp.INTENDED_PROBABILITY,"perpendicular_slip_probability":mdp.SLIP_PROBABILITY},"rewards":{"step":rw.step,"collision_extra":rw.collision,"penalty_extra":rw.penalty,"first_key":rw.key,"goal":rw.goal,"shaping_scale":rw.shaping_scale},"shaping":{"enabled":config.reward_mode=="shaped","method":config.shaping_method,"version":config.shaping_version,"scale":rw.shaping_scale},"learning":{"gamma":config.gamma,"alpha":config.alpha,"lambda":config.trace_lambda,"episodes":config.episodes,"epsilon_start":config.epsilon_start,"epsilon_end":config.epsilon_end,"decay_episodes":config.decay_episodes,"schedule":config.schedule,"reward_mode":config.reward_mode,"diagnostic_episode":config.diagnostic_episode},"actions":{"order":list(ACTION_NAMES)},"seeds":{"root":seeds.root,"behavior":seeds.behavior,"transition":seeds.transition,"derivation":SEED_DERIVATION},"behavior_policy":{"identifier":BEHAVIOR_POLICY},"q_initialization":{"identifier":Q_INITIALIZATION},"traces":{"type":TRACE_TYPE,"update_order":TRACE_UPDATE_ORDER,"reset":TRACE_RESET},"episode_semantics":{"termination":"goal_no_bootstrap","truncation":TRUNCATION_SEMANTICS}}
    text=_canonical_json(resolved); digest=hashlib.sha256(text.encode()).hexdigest()
    return SarsaLambdaRunIdentity(text,digest,f"sarsa-lambda-{digest}")

def apply_sarsa_lambda_update(q_values,traces,*,state,intended_action,reward,next_state,next_action,alpha,gamma,trace_lambda,terminated,truncated,active_mask=None):
    if terminated and truncated: raise ValueError("transition cannot terminate and truncate")
    action=Action.parse(intended_action); ai=ACTION_INDEX[action]; idx=(*state_index(state),ai)
    old=float(q_values[idx]); next_ai=None if next_action is None else ACTION_INDEX[Action.parse(next_action)]
    if terminated:
        next_q=bootstrap=0.0
    else:
        if next_ai is None: raise ValueError("continuing/truncated update requires next action")
        next_q=float(q_values[(*state_index(next_state),next_ai)]); bootstrap=gamma*next_q
    target=float(reward)+bootstrap; delta=target-old
    traces[idx]=1.0
    active=(traces!=0) if active_mask is None else ((traces!=0)&active_mask)
    q_values[active] += alpha*delta*traces[active]
    count=int(active.sum())
    if terminated or truncated: traces.fill(0.0)
    else: traces[active] *= gamma*trace_lambda
    return TraceUpdate(state,action,ai,float(reward),next_state,None if next_ai is None else ACTION_ORDER[next_ai],next_ai,old,next_q,bootstrap,target,delta,count,terminated,truncated)

def train_sarsa_lambda(mdp,config,*,root_seed,identity=None):
    if config.reward_mode=="shaped" and (not mdp.use_shaping or mdp.gamma!=config.gamma): raise ValueError("shaped MDP must enable shaping with matching gamma")
    valid=valid_state_mask(mdp.spec); reachable=reachable_state_mask(mdp); terminal=terminal_state_mask(mdp.spec)
    q=dense_q_array(mdp.spec); q[valid]=0.; q[terminal]=0.; counts=np.zeros(valid.shape,np.int64); acounts=np.zeros(q.shape,np.int64)
    active_mask=np.broadcast_to((valid&~terminal)[...,None],q.shape)
    seeds=derive_sarsa_lambda_seeds(root_seed); rid=build_sarsa_lambda_run_identity(mdp,config,seeds)
    if identity is not None and identity != rid: raise ValueError("precomputed identity mismatch")
    rng=np.random.default_rng(seeds.behavior); env=MazeEpisode(mdp,seed=seeds.transition); metrics=[]; diagnostics=[]; started=time.perf_counter()
    for ep in range(1,config.episodes+1):
        epstart=time.perf_counter(); epsilon=epsilon_for_episode(config,ep); traces=np.zeros(q.shape,np.float64)
        state=env.reset(); counts[state_index(state)]+=1; visited={state}; events=Counter(); base=shaping=0.
        selection=select_epsilon_greedy(q[state_index(state)],epsilon,rng)
        while not env.done:
            source=state; acounts[(*state_index(source),selection.action_index)]+=1; step=env.step(selection.action)
            learning=step.base_reward if config.reward_mode=="sparse" else step.total_reward
            next_selection=None if step.terminated else select_epsilon_greedy(q[state_index(step.state)],epsilon,rng)
            snapshot=[]
            if ep==config.diagnostic_episode:
                traces[(*state_index(source),selection.action_index)]=1.
                for ti in zip(*np.nonzero(traces),strict=True): snapshot.append((ti,float(traces[ti]),float(q[ti])))
                traces[(*state_index(source),selection.action_index)]=0. if all(t[0]!=(*state_index(source),selection.action_index) for t in snapshot) else traces[(*state_index(source),selection.action_index)]
            update=apply_sarsa_lambda_update(q,traces,state=source,intended_action=selection.action,reward=learning,next_state=step.state,next_action=None if next_selection is None else next_selection.action,alpha=config.alpha,gamma=config.gamma,trace_lambda=config.trace_lambda,terminated=step.terminated,truncated=step.truncated,active_mask=active_mask)
            if ep==config.diagnostic_episode:
                reset=step.terminated or step.truncated
                for ti,e,oldq in snapshot:
                    inc=e*update.td_error*config.alpha
                    diagnostics.append(TraceDiagnosticRow(ep,step.step_number,epsilon,selection.exploring,int(source.has_key),source.row,source.col,selection.action.name,selection.action_index,step.actual_action.name,ACTION_INDEX[step.actual_action],step.probability,int(step.state.has_key),step.state.row,step.state.col,"|".join(x.value for x in step.events),step.base_reward,step.shaping_reward,step.total_reward,learning,"" if next_selection is None else next_selection.action.name,-1 if next_selection is None else next_selection.action_index,False if next_selection is None else next_selection.exploring,update.bootstrap_value,update.target,update.td_error,ti[0],ti[1],ti[2],ACTION_ORDER[ti[3]].name,ti[3],e,oldq,inc,oldq+inc,0. if reset else config.gamma*config.trace_lambda*e,step.terminated,step.truncated,reset,config.alpha,config.gamma,config.trace_lambda))
            state=step.state; counts[state_index(state)]+=1; visited.add(state); base+=step.base_reward; shaping+=step.shaping_reward; events.update(step.events)
            if not env.done: selection=next_selection
        if np.any(traces): raise ValueError("eligibility traces leaked across episode boundary")
        metrics.append(EpisodeMetrics(ep,epsilon,env.elapsed_steps,base,shaping,base+shaping,base if config.reward_mode=="sparse" else base+shaping,bool(events[EventType.GOAL_REACHED]),bool(events[EventType.GOAL_REACHED]),bool(events[EventType.EPISODE_TRUNCATED]),int(events[EventType.MOVE]),int(events[EventType.WALL_COLLISION]),int(events[EventType.PENALTY_ENTERED]),int(events[EventType.KEY_COLLECTED]),int(events[EventType.CLOSED_DOOR_ATTEMPT]),int(events[EventType.DOOR_PASSED]),int(events[EventType.TELEPORTED]),int(events[EventType.GOAL_REACHED]),int(events[EventType.EPISODE_TRUNCATED]),len(visited),env.elapsed_steps+1-len(visited),time.perf_counter()-epstart))
    return SarsaLambdaResult(q,counts,acounts,valid,reachable,terminal,tuple(metrics),tuple(diagnostics),config,seeds,rid,time.perf_counter()-started,map_checksum(mdp.spec))

def _csv(path,result,rows,row_type):
    path.parent.mkdir(parents=True,exist_ok=True); provenance={k:result.metadata()[k] for k in ("run_id","semantic_config_hash","map_checksum","lambda","reward_mode","schedule","root_seed")}; names=[f.name for f in fields(row_type)]
    with path.open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=[*provenance,*names]); w.writeheader()
        for row in rows: w.writerow({**provenance,**{n:(repr(getattr(row,n)) if isinstance(getattr(row,n),float) else int(getattr(row,n)) if isinstance(getattr(row,n),bool) else getattr(row,n)) for n in names}})
def preflight_sarsa_lambda_bundle(paths,*,overwrite=False):
    existing=[p for p in paths.all_paths() if p.exists()]
    if existing and not overwrite: raise FileExistsError("Refusing to overwrite existing artifact(s): "+", ".join(map(str,existing)))
def _hash(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save_sarsa_lambda_bundle(paths,result,*,expected_spec,overwrite=False):
    preflight_sarsa_lambda_bundle(paths,overwrite=overwrite)
    for p in paths.all_paths(): p.parent.mkdir(parents=True,exist_ok=True)
    token=uuid.uuid4().hex; staged=SarsaLambdaBundlePaths(*(p.with_name(f".{p.stem}.{token}.tmp{p.suffix}") for p in paths.all_paths())); published=[]
    try:
        save_sarsa_lambda_npz(staged.model,q_values=result.q_values,state_visit_counts=result.state_visit_counts,state_action_visit_counts=result.state_action_visit_counts,valid_mask=result.valid_state_mask,reachable_mask=result.reachable_state_mask,terminal_mask=result.terminal_state_mask,metadata=result.metadata())
        _csv(staged.episode_metrics,result,result.episode_metrics,EpisodeMetrics); _csv(staged.diagnostic,result,result.diagnostic_rows,TraceDiagnosticRow)
        load_sarsa_lambda_npz(staged.model,expected_spec=expected_spec)
        doc={"manifest_schema_version":1,"csv_schema_version":1,"algorithm":ALGORITHM_ID,"run_id":result.identity.run_id,"semantic_config_hash":result.identity.semantic_config_hash,"run_config_json":result.identity.config_json,"complete":True,"artifacts":{}}
        for name,temp,final,rows in (("model",staged.model,paths.model,None),("episode_metrics",staged.episode_metrics,paths.episode_metrics,len(result.episode_metrics)),("diagnostic",staged.diagnostic,paths.diagnostic,len(result.diagnostic_rows))): doc["artifacts"][name]={"path":Path(os.path.relpath(final,paths.manifest.parent)).as_posix(),"sha256":_hash(temp),"bytes":temp.stat().st_size,**({} if rows is None else {"row_count":rows})}
        staged.manifest.write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        for temp,final in zip(staged.all_paths(),paths.all_paths()): os.replace(temp,final); published.append(final)
        loaded,manifest,_=validate_sarsa_lambda_bundle(paths.manifest,expected_spec=expected_spec,expected_model=paths.model); return loaded,manifest
    except Exception:
        for p in published: p.unlink(missing_ok=True)
        raise
    finally:
        for p in staged.all_paths(): p.unlink(missing_ok=True)
def validate_sarsa_lambda_bundle(manifest_path,*,expected_spec=None,expected_model=None):
    mp=Path(manifest_path).resolve()
    try: doc=json.loads(mp.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError) as e: raise ValueError(f"Incomplete SARSA(lambda) bundle: {e}") from e
    if doc.get("complete") is not True or doc.get("algorithm")!=ALGORITHM_ID or doc.get("manifest_schema_version")!=1: raise ValueError("invalid SARSA(lambda) manifest")
    arts=doc.get("artifacts",{}); resolved={}
    for name in ("model","episode_metrics","diagnostic"):
        entry=arts.get(name);
        if not isinstance(entry,dict): raise ValueError(f"missing SARSA artifact {name}")
        p=(mp.parent/entry["path"]).resolve(); resolved[name]=p
        if not p.exists() or _hash(p)!=entry.get("sha256") or p.stat().st_size!=entry.get("bytes"): raise ValueError(f"SARSA {name} artifact validation failed")
    paths=SarsaLambdaBundlePaths(resolved["model"],resolved["episode_metrics"],resolved["diagnostic"],mp)
    if expected_model is not None and paths.model!=Path(expected_model).resolve(): raise ValueError("manifest references different model")
    loaded=load_sarsa_lambda_npz(paths.model,expected_spec=expected_spec)
    if doc["run_id"]!=loaded.metadata["run_id"] or doc["semantic_config_hash"]!=loaded.metadata["semantic_config_hash"] or doc["run_config_json"]!=loaded.metadata["run_config_json"]: raise ValueError("manifest/model provenance mismatch")
    for name,path in (("episode_metrics",paths.episode_metrics),("diagnostic",paths.diagnostic)):
        with path.open(encoding="utf-8",newline="") as h: rows=list(csv.DictReader(h))
        if len(rows)!=arts[name]["row_count"] or any(r["run_id"]!=doc["run_id"] for r in rows): raise ValueError(f"SARSA {name} CSV validation failed")
        if name=="diagnostic":
            for r in rows:
                if float(r["new_q"]) != float(r["old_q"])+float(r["eligibility"])*float(r["td_error"])*float(r["alpha"]): raise ValueError("diagnostic reconstruction failed")
    return loaded,doc,paths
