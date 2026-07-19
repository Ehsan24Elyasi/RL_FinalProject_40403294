from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest

from agents.common import ACTION_ORDER, dense_q_array, load_sarsa_lambda_npz
from agents.sarsa_lambda import (SarsaLambdaBundlePaths, SarsaLambdaConfig,
    apply_sarsa_lambda_update, build_sarsa_lambda_run_identity,
    derive_sarsa_lambda_seeds, save_sarsa_lambda_bundle, train_sarsa_lambda,
    validate_sarsa_lambda_bundle)
from config import load_config
from environments.generator import load_source_map
from environments.maze import Action, MazeMDP, State
from main import main

ROOT=Path(__file__).parents[1]

def arrays():
    spec=load_source_map(ROOT/'environments/maps/source.json')
    q=dense_q_array(spec); valid=np.ones(q.shape[:-1],bool)
    for r,c in spec.walls: valid[:,r,c]=False
    q[valid]=0.; traces=np.zeros_like(q)
    return spec,q,traces,np.broadcast_to(valid[...,None],q.shape)

def test_replacing_trace_decay_and_multi_entry_update():
    _,q,e,mask=arrays(); s=State(1,1,False); prior=State(1,2,False)
    e[(0,1,2,1)]=.5
    u=apply_sarsa_lambda_update(q,e,state=s,intended_action=Action.RIGHT,reward=1,next_state=prior,next_action=Action.DOWN,alpha=.1,gamma=.9,trace_lambda=.5,terminated=False,truncated=False,active_mask=mask)
    assert u.td_error==1
    assert q[(0,1,1,1)]==pytest.approx(.1) and q[(0,1,2,1)]==pytest.approx(.05)
    assert e[(0,1,1,1)]==pytest.approx(.45) and e[(0,1,2,1)]==pytest.approx(.225)
    apply_sarsa_lambda_update(q,e,state=s,intended_action=Action.RIGHT,reward=0,next_state=prior,next_action=Action.DOWN,alpha=.1,gamma=.9,trace_lambda=.5,terminated=False,truncated=False,active_mask=mask)
    assert e[(0,1,1,1)]==pytest.approx(.45)  # replaced, then decayed

def test_terminal_and_truncation_targets_clear_traces():
    _,q,e,mask=arrays(); s=State(1,1,False); n=State(1,2,False)
    q[(0,1,2,2)]=4
    terminal=apply_sarsa_lambda_update(q,e,state=s,intended_action=Action.RIGHT,reward=2,next_state=n,next_action=None,alpha=.1,gamma=.9,trace_lambda=.7,terminated=True,truncated=False,active_mask=mask)
    assert terminal.target==2 and not e.any()
    q[(0,1,1,1)]=0
    truncated=apply_sarsa_lambda_update(q,e,state=s,intended_action=Action.RIGHT,reward=2,next_state=n,next_action=Action.DOWN,alpha=.1,gamma=.9,trace_lambda=.7,terminated=False,truncated=True,active_mask=mask)
    assert truncated.target==pytest.approx(5.6) and not e.any()

def test_lambda_zero_is_one_step_sarsa():
    _,q,e,mask=arrays(); s=State(1,1,False); n=State(1,2,False); q[(0,1,2,2)]=3
    apply_sarsa_lambda_update(q,e,state=s,intended_action=Action.RIGHT,reward=-1,next_state=n,next_action=Action.DOWN,alpha=.2,gamma=.9,trace_lambda=0,terminated=False,truncated=False,active_mask=mask)
    assert q[(0,1,1,1)]==pytest.approx(.2*(-1+.9*3)) and not e.any()

def test_config_required_lambda_order():
    cfg=load_config(ROOT/'config.yaml')
    assert tuple(r.trace_lambda for r in cfg.sarsa_lambda.required_runs)==(0,.3,.7,.9)

def short_result(lam=.3,episodes=2):
    cfg=load_config(ROOT/'config.yaml'); spec=load_source_map(cfg.source_map)
    mdp=MazeMDP(spec,cfg.rewards,gamma=.95,use_shaping=True)
    sc=SarsaLambdaConfig(trace_lambda=lam,episodes=episodes,decay_episodes=2,diagnostic_episode=1)
    seeds=derive_sarsa_lambda_seeds(9); identity=build_sarsa_lambda_run_identity(mdp,sc,seeds)
    return spec,train_sarsa_lambda(mdp,sc,root_seed=9,identity=identity)

def test_fixed_seed_reproducibility():
    _,a=short_result(0,1); _,b=short_result(0,1)
    assert np.array_equal(a.q_values,b.q_values,equal_nan=True)
    assert np.array_equal(a.state_action_visit_counts,b.state_action_visit_counts)
    assert [replace(x,runtime_seconds=0) for x in a.episode_metrics]==[replace(x,runtime_seconds=0) for x in b.episode_metrics]
    assert a.diagnostic_rows==b.diagnostic_rows

def test_npz_bundle_roundtrip_and_corruption(tmp_path):
    spec,result=short_result(.7,1); p=SarsaLambdaBundlePaths(tmp_path/'m.npz',tmp_path/'e.csv',tmp_path/'d.csv',tmp_path/'manifest.json')
    loaded,_=save_sarsa_lambda_bundle(p,result,expected_spec=spec)
    assert np.array_equal(loaded.q_values,result.q_values,equal_nan=True)
    validate_sarsa_lambda_bundle(p.manifest,expected_spec=spec,expected_model=p.model)
    p.diagnostic.write_text(p.diagnostic.read_text()+'x',encoding='utf-8')
    with pytest.raises(ValueError): validate_sarsa_lambda_bundle(p.manifest,expected_spec=spec)

def test_cli_train_and_inspect(tmp_path,capsys):
    assert main(['sarsa','train','--lambda','0.9','--reward-mode','shaped','--episodes','1','--decay-episodes','2','--diagnostic-episode','1','--output-dir',str(tmp_path)])==0
    model=next((tmp_path/'models').glob('*.npz'))
    assert main(['sarsa','inspect',str(model)])==0
    assert 'trace=replacing' in capsys.readouterr().out

@pytest.mark.parametrize('lam',[0,.3,.7,.9])
def test_all_lambdas_canonical_smoke(lam):
    _,result=short_result(lam,1)
    assert len(result.episode_metrics)==1 and result.episode_metrics[0].steps>0
    assert not np.any(result.state_action_visit_counts[result.terminal_state_mask])
