import argparse
from rich import print,console
from pathlib import Path
import os
import hydra
import ray

from minestudio.simulator import MinecraftSim
from minestudio.simulator.entry import CameraConfig
from minestudio.simulator.callbacks import (
    SpeedTestCallback, 
    RecordCallback2, 
    RewardsCallback, 
    TaskCallback, 
    FastResetCallback2, 
    InitInventoryCallback,
    SummonMobsCallback,
    CommandsCallback,
    TeleportCallback,
)
from minestudio.models import CraftWorker,SmeltWorker

from actvlp.evaluate import draw_utils
from actvlp.utils import file_utils
from actvlp.evaluate import agent_wrapper


def evaluate(video_path,checkpoints,environment_config:dict,model_config:dict,device="cuda:0",base_url=None):

    hydra.core.global_hydra.GlobalHydra.instance().clear() # 清理 Hydra 的全局实例
    config_path = Path(f"{environment_config['env_config']}.yaml")
    config_name = config_path.stem
    config_path = os.path.join("./config",config_path.parent)
    hydra.initialize(config_path=config_path, version_base='1.3')
    cfg = hydra.compose(config_name=config_name)
    
    # camera_config
    camera_cfg = CameraConfig(**cfg.camera_config)
    record_callback = RecordCallback2(record_path=Path(video_path).parent, fps=30,show_actions=True)  
    callbacks = [
        FastResetCallback2(
            biomes=cfg.candidate_preferred_spawn_biome,
            random_tp_range=cfg.random_tp_range,
            start_time=cfg.start_time,
        ), 
        SpeedTestCallback(50), 
        TaskCallback(getattr(cfg,"task_conf",None)),
        RewardsCallback(getattr(cfg,"reward_conf",None)),
        InitInventoryCallback(cfg.init_inventory,
                                inventory_distraction_level=cfg.inventory_distraction_level,
                                equip_distraction_level="normal"
                                ),
        CommandsCallback(getattr(cfg,"command",[]),),
        record_callback,
    ]
    if hasattr(cfg,"teleport"):
        callbacks.append(TeleportCallback(x=cfg.teleport.x, y=cfg.teleport.y, z=cfg.teleport.z,))
    if cfg.mobs:
        callbacks.append(SummonMobsCallback(cfg.mobs))
    
    # init env
    env =  MinecraftSim(
        action_type="env",
        seed=cfg.seed,
        obs_size=cfg.origin_resolution,
        render_size=cfg.resize_resolution,
        camera_config=camera_cfg,
        preferred_spawn_biome=getattr(cfg,"preferred_spawn_biome",None),
        callbacks = callbacks
    )
    obs, info = env.reset()

    # init agent
    agent = None
    pre_agent = None
    worker_type =  getattr(cfg,"worker", None)
    if worker_type == "craft":
        pre_agent = CraftWorker(env,if_discrete=True)
    elif worker_type == "smelt":
        pre_agent = SmeltWorker(env,if_discrete=True)
    
    # 把环境准备好
    need_crafting_table = False
    if getattr(cfg, "need_gui", False):
        need_crafting_table= getattr(cfg,"need_crafting_table", False)
        need_furnace = getattr(cfg,"need_furnace", False)
        if need_crafting_table:
            try:
                frames,_,_ = pre_agent.open_crating_table_wo_recipe()
            except AssertionError as e:
                env.close()
                console.Console().log(f"error: {e}")
                return False,-1
        elif need_furnace:
            try:
                frames,_,_ = pre_agent.open_furnace_wo_recipe()
            except AssertionError as e:
                env.close()
                console.Console().log(f"error: {e}")
                return False,-1
        else:
            pre_agent._null_action(1)
            if not pre_agent.info['isGuiOpen']:
                pre_agent._call_func('inventory')
        # turn env 
        
    env.action_type = "agent"  
    record_callback.forget()

    if type(base_url)!=type(None):
        agent = agent_wrapper.VLLM_AGENT(checkpoint_path=checkpoints,base_url=base_url,**model_config)
    else:
        raise Exception
        
    # get instruction
    instructions = [item["text"] for item in cfg.task_conf]

    success = (False,environment_config["max_frames"])
    for i in range(environment_config["max_frames"]):
        action = agent.forward([info["pov"]],instructions,verbos=environment_config["verbos"],need_crafting_table = need_crafting_table)
        if environment_config["verbos"]:
            console.Console().log(action)
        obs, reward, terminated, truncated, info = env.step(action)

        if reward>0:
            success = (True,i)
            break   
        
    # sample another 30 steps if success
    if success[0]:
        for i in range(20):
            action = agent.forward([info["pov"]],instructions,verbos=environment_config["verbos"],need_crafting_table = need_crafting_table)
            obs, reward, terminated, truncated, info = env.step(action)
         
    env.close()
    return success

@ray.remote
def evaluate_wrapper(video_path,checkpoints,environment_config,base_url,model_config):

    success = evaluate(video_path=video_path,checkpoints=checkpoints,environment_config=environment_config,base_url=base_url,model_config=model_config)
    member_id = video_path.split("/")[-1].split(".")[0]
    return success[0],success[1],member_id

def multi_evaluate(args):
    ray.init()
    import os
    from pathlib import Path
    
    model_ref_name = args.checkpoints.split('/')[-1]
    if "checkpoint" in model_ref_name:
        checkpoint_num = model_ref_name.split("-")[-1]
        model_base_name = args.checkpoints.split('/')[-2]
        model_ref_name = f"{model_base_name}-{checkpoint_num}"
    
    video_fold  = os.path.join(args.video_main_fold, f"{model_ref_name}-{args.env_config.split('/')[-1]}") 
    if not os.path.exists(video_fold):
        Path(video_fold).mkdir(parents=True,exist_ok=True)
    
    model_config = dict(
        temperature=args.temperature,
        history_num = args.history_num,
        instruction_type = args.instruction_type,
        action_chunk_len = args.action_chunk_len,
    )
    environment_config = dict(
        env_config = args.env_config,
        max_frames = args.max_frames,
        verbos = args.verbos,
    )
    
    video_log_path = os.path.join(video_fold,"end.json") 
    resultss = file_utils.load_json_file(video_log_path,data_type="list")

    total_ids = [i for i in range(args.workers)]
    done_ids = [results[2] for results in resultss]
    undone_ids = [id for id in total_ids if str(id) not in done_ids]

    if not undone_ids:
        return
    
    roll = len(undone_ids) // args.split_number + (1 if len(undone_ids) % args.split_number != 0 else 0)
    for i in range(roll):
        part_undone_ids = undone_ids[i*args.split_number:min((i+1)*args.split_number, len(undone_ids))]
        result_ids = [evaluate_wrapper.remote(video_path=os.path.join(video_fold,str(i),f"{i}.mp4"),checkpoints=args.checkpoints,environment_config=environment_config,base_url=args.base_url,model_config=model_config) for i in part_undone_ids]
        futures = result_ids
        
        while len(futures) > 0:
            ready_futures, rest_futures = ray.wait(futures,timeout=24*60*60)
            results = ray.get(ready_futures,timeout=60*60)  # Retrieve all results
            resultss.extend(results)
            print(f"part frames IDs: {results} done!")
            futures = rest_futures
        
        ray.shutdown()
        
        # 写入日志文件
        file_utils.dump_json_file(resultss,video_log_path)
    draw_utils.show_success_rate(resultss,os.path.join(video_fold,"image.png") )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=1) 
    parser.add_argument('--split-number', type=int, default=6) 
    parser.add_argument('--env-config',"-e", type=str, default='craft/craft_bread') #vpt/test_vpt
    parser.add_argument('--max-frames', type=int, default=200) #vpt/test_vpt
    parser.add_argument('--verbos', type=bool, default=False)
    parser.add_argument('--checkpoints', type=str, default="/public/models/qwen2-vl-7b-instruct/")
    parser.add_argument('--device',type=str,default="cuda:1")
    
    parser.add_argument('--base-url',type=str)
    parser.add_argument('--video-main-fold',type=str)
    
    parser.add_argument('--instruction-type',type=str,default='normal')
    parser.add_argument('--temperature','-t',type=float,default=0.7)
    parser.add_argument('--history-num',type=int,default=0)
    parser.add_argument('--action-chunk-len',type=int,default=1)

    args = parser.parse_args()

    model_config = dict(
        temperature=args.temperature,
        history_num = args.history_num,
        instruction_type = args.instruction_type,
        action_chunk_len = args.action_chunk_len,
    )
    environment_config = dict(
        env_config = args.env_config,
        max_frames = args.max_frames,
        verbos = args.verbos,
    )
    if not args.base_url:
        args.base_url=None
    
    if args.workers==0:
        environment_config["verbos"] = True
        video_path = f"{args.checkpoints.split('/')[-1]}-{args.env_config.split('/')[-1]}.mp4"
        evaluate(video_path=video_path,checkpoints = args.checkpoints,environment_config = environment_config,device=args.device,model_config=model_config)
    elif args.workers==1:
        video_path = f"{args.checkpoints.split('/')[-1]}-{args.env_config.split('/')[-1]}.mp4"
        evaluate(video_path=f"{args.checkpoints.split('/')[-1]}-{args.env_config.split('/')[-1]}.mp4",checkpoints = args.checkpoints,environment_config = environment_config,base_url=args.base_url,model_config=model_config)
    elif args.workers>1:
        multi_evaluate(args)
        
    
