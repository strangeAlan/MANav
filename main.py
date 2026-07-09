import os
import sys
import json
import logging
import time
import yaml
from collections import deque, defaultdict
from types import SimpleNamespace
import numpy as np
import torch
import argparse
from src.envs import construct_envs
from src.agent.unigoal.agent import UniGoal_Agent
from src.map.bev_mapping import BEV_Map
from src.graph.graph import Graph
from src.memory.sparse_scene_memory import SparseSceneMemory
from src.perception.lingbot_depth_client import LingBotDepthClient
import gzip

def get_config():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="configs/config_habitat.yaml",
                        metavar="FILE", help="path to config file", type=str)
    parser.add_argument("--goal_type", default="ins-image", type=str)
    parser.add_argument("--episode_id", default=-1, type=int, help="episode id, 0~999")
    parser.add_argument("--goal", default="", type=str)
    parser.add_argument("--real_world", action="store_true")

    args = parser.parse_args()

    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    args = vars(args)
    args.update(config)
        
    args = SimpleNamespace(**args)

    args.is_debugging = sys.gettrace() is not None
    if args.is_debugging:
        args.experiment_id = "debug"
    
    args.log_dir = os.path.join(args.dump_location, args.experiment_id, 'log')
    args.visualization_dir = os.path.join(args.dump_location, args.experiment_id, 'visualization')

    args.map_size = args.map_size_cm // args.map_resolution
    args.global_width, args.global_height = args.map_size, args.map_size
    args.local_width = int(args.global_width / args.global_downscaling)
    args.local_height = int(args.global_height / args.global_downscaling)

    args.device = torch.device("cuda:0" if args.cuda else "cpu")

    args.num_scenes = args.num_processes
    args.num_episodes = int(args.num_eval_episodes)

    return args


def main():
    args = get_config()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.visualization_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(args.log_dir, 'eval.log'),
        level=logging.INFO)
    logging.info(args)

    eval_metrics_id = 0

    episode_success = deque(maxlen=args.num_episodes)
    episode_spl = deque(maxlen=args.num_episodes)

    finished = False
    wait_env = False

    if args.goal_type == 'text':
        with gzip.open(args.text_goal_dataset, 'rt') as f:
            text_goal_dataset = json.load(f)

    BEV_map = BEV_Map(args)
    graph = Graph(args)
    envs = construct_envs(args)
    agent = UniGoal_Agent(args, envs)
    sparse_memory = None
    sparse_depth_client = None
    if getattr(args, "sparse_lingbot_graph", False):
        sparse_memory = SparseSceneMemory(
            args,
            llm=graph.llm if getattr(args, "sparse_memory_use_vlm_hint", False) else None,
        )
        if getattr(args, "sparse_memory_use_lingbot_depth", True):
            sparse_depth_client = LingBotDepthClient(
                base_url=getattr(args, "lingbot_depth_url", "http://127.0.0.1:18180"),
                timeout=float(getattr(args, "lingbot_depth_timeout", 180.0)),
                min_depth_m=float(getattr(args, "lingbot_min_depth_m", 0.2)),
                max_depth_m=float(getattr(args, "lingbot_max_depth_m", args.max_depth)),
                scale=float(getattr(args, "lingbot_depth_scale", 1.0)),
                confidence_threshold=float(getattr(args, "lingbot_confidence_threshold", 0.0)),
                invalid_fill_m=float(getattr(args, "lingbot_invalid_fill_m", args.max_depth)),
                max_jump_m=float(getattr(args, "lingbot_max_jump_m", 0.0)),
                temporal_alpha=float(getattr(args, "lingbot_temporal_alpha", 0.0)),
            )

    BEV_map.init_map_and_pose()
    obs, rgbd, infos = agent.reset()
    if sparse_memory is not None:
        sparse_memory.reset(infos.get("goal_name", ""))
        if sparse_depth_client is not None:
            sparse_depth_client.reset()

    BEV_map.mapping(rgbd, infos)

    global_goals = [args.local_width // 2, args.local_height // 2]

    goal_maps = np.zeros((args.local_width, args.local_height))

    goal_maps[global_goals[0], global_goals[1]] = 1

    agent_input = {}
    agent_input['map_pred'] = BEV_map.local_map[0, 0, :, :].cpu().numpy()
    agent_input['exp_pred'] = BEV_map.local_map[0, 1, :, :].cpu().numpy()
    agent_input['pose_pred'] = BEV_map.planner_pose_inputs[0]
    agent_input['goal'] = goal_maps
    agent_input['exp_goal'] = goal_maps * 1
    agent_input['new_goal'] = 1
    agent_input['found_goal'] = 0
    agent_input['wait'] = wait_env or finished
    agent_input['sem_map'] = BEV_map.local_map[0, 4:11, :, :
                                        ].cpu().numpy()
    if args.visualize:
        BEV_map.local_map[0, 10, :, :] = 1e-5
        agent_input['sem_map_pred'] = BEV_map.local_map[0, 4:11, :, :
                                            ].argmax(0).cpu().numpy()

    obs, rgbd, done, infos = agent.step(agent_input)

    graph.reset()
    graph.set_obj_goal(infos['goal_name'])
    if args.goal_type == 'ins-image':
        graph.set_image_goal(infos['instance_imagegoal'])
    elif args.goal_type == 'text':
        graph.set_text_goal(infos['text_goal'])

    step = 0

    while True:
        if finished == True:
            break

        global_step = (step // args.num_local_steps) % args.num_global_steps
        local_step = step % args.num_local_steps

        if done:
            spl = infos['spl']
            success = infos['success']
            success = success if success is not None else 0.0
            eval_metrics_id += 1
            episode_success.append(success)
            episode_spl.append(spl)
            if len(episode_success) == args.num_episodes:
                finished = True
            if args.visualize:
                video_path = os.path.join(args.visualization_dir, 'videos', 'eps_{:0>6}.mp4'.format(infos['episode_no']))
                agent.save_visualization(video_path)
            wait_env = True
            BEV_map.update_intrinsic_rew()
            BEV_map.init_map_and_pose_for_env()

            graph.reset()
            graph.set_obj_goal(infos['goal_name'])
            if args.goal_type == 'ins-image':
                graph.set_image_goal(infos['instance_imagegoal'])
            elif args.goal_type == 'text':
                graph.set_text_goal(infos['text_goal'])
            if sparse_memory is not None:
                sparse_memory.reset(infos.get("goal_name", ""))
                if sparse_depth_client is not None:
                    sparse_depth_client.reset()

        BEV_map.mapping(rgbd, infos)
        if sparse_memory is not None and not wait_env:
            sparse_memory.update_pose(step, BEV_map.full_pose[0].detach().cpu().numpy(), args.map_resolution)
            update_interval = int(getattr(args, "sparse_memory_update_interval", 5))
            if update_interval <= 0 or step % update_interval == 0:
                depth_result = None
                if sparse_depth_client is not None and getattr(agent, "raw_obs", None) is not None:
                    try:
                        depth_result = sparse_depth_client.predict(agent.raw_obs.astype(np.uint8))
                    except Exception as exc:
                        logging.info("[SparseMemory] LingBot side depth failed at step %s: %s", step, exc)
                sparse_memory.update_observation(
                    step=step,
                    rgb=getattr(agent, "raw_obs", None),
                    detections=getattr(agent, "pred_box", []),
                    depth_m=None if depth_result is None else depth_result.depth_m,
                    confidence=None if depth_result is None else depth_result.confidence,
                )
                sparse_memory.maybe_update_vlm_hint(step)
                if getattr(args, "sparse_memory_log", False):
                    logging.info(
                        "[SparseMemory] step=%s stable=%s preferred=%s",
                        step,
                        sparse_memory.memory_prompt_items()[:8],
                        sparse_memory.vlm_preferred_labels,
                    )

        navigate_steps = global_step * args.num_local_steps + local_step
        graph.set_navigate_steps(navigate_steps)
        if not agent_input['wait'] and navigate_steps % 2 == 0:
            graph.set_observations(obs)
            graph.update_scenegraph()

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        if local_step == args.num_local_steps - 1 or np.linalg.norm(np.array([BEV_map.local_row, BEV_map.local_col]) - np.array(global_goals)) < 10:
            if wait_env == True:
                wait_env = False
            else:
                BEV_map.update_intrinsic_rew()

            BEV_map.move_local_map()

            graph.set_full_map(BEV_map.full_map)
            graph.set_full_pose(BEV_map.full_pose)
            goal = graph.explore()
            if (
                sparse_memory is not None
                and getattr(args, "sparse_memory_frontier_bias", False)
                and hasattr(graph, "frontier_locations_16")
            ):
                biased_goal = sparse_memory.choose_frontier(
                    graph.frontier_locations_16,
                    BEV_map.full_pose[0].detach().cpu().numpy(),
                    args.map_resolution,
                    default_goal=goal,
                )
                if biased_goal is not None:
                    if getattr(args, "sparse_memory_log", False):
                        message = "[SparseMemory] step={} frontier_bias goal={} -> {}".format(
                            step,
                            goal.tolist() if hasattr(goal, "tolist") else goal,
                            biased_goal.tolist() if hasattr(biased_goal, "tolist") else biased_goal,
                        )
                        print(message)
                        logging.info(message)
                    goal = biased_goal
            if hasattr(graph, 'frontier_locations_16'):
                graph.frontier_locations_16[:, 0] = graph.frontier_locations_16[:, 0] - BEV_map.local_map_boundary[0, 0]
                graph.frontier_locations_16[:, 1] = graph.frontier_locations_16[:, 1] - BEV_map.local_map_boundary[0, 2]
            if isinstance(goal, list) or isinstance(goal, np.ndarray):
                goal = list(goal)
                goal[0] = goal[0] - BEV_map.local_map_boundary[0, 0]
                goal[1] = goal[1] - BEV_map.local_map_boundary[0, 2]
                if 0 <= goal[0] < args.local_width and 0 <= goal[1] < args.local_height:
                    global_goals = goal


        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        found_goal = False
        goal_maps = np.zeros((args.local_width, args.local_height))

        goal_maps[global_goals[0], global_goals[1]] = 1

        exp_goal_maps = goal_maps.copy()

        agent_input = {}
        agent_input['map_pred'] = BEV_map.local_map[0, 0, :, :].cpu().numpy()
        agent_input['exp_pred'] = BEV_map.local_map[0, 1, :, :].cpu().numpy()
        agent_input['pose_pred'] = BEV_map.planner_pose_inputs[0]
        agent_input['goal'] = goal_maps
        agent_input['exp_goal'] = exp_goal_maps
        agent_input['new_goal'] = local_step == args.num_local_steps - 1
        agent_input['found_goal'] = found_goal
        agent_input['wait'] = wait_env or finished
        agent_input['sem_map'] = BEV_map.local_map[0, 4:11, :, :
                                        ].cpu().numpy()

        if args.visualize:
            BEV_map.local_map[0, 10, :, :] = 1e-5
            agent_input['sem_map_pred'] = BEV_map.local_map[0, 4:11, :,
                                                :].argmax(0).cpu().numpy()

        obs, rgbd, done, infos = agent.step(agent_input)

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # log
        if step % args.log_interval == 0:
            log = " ".join([
                "num timesteps {},".format(step),
                "episode_id {}".format(infos['episode_no']),
            ])

            total_success = []
            total_spl = []
            for acc in episode_success:
                total_success.append(acc)
            for spl in episode_spl:
                total_spl.append(spl)

            if len(total_spl) > 0:
                log += " Average SR/SPL:"
                log += " {:.5f}/{:.5f},".format(
                    np.mean(total_success),
                    np.mean(total_spl))

            print(log)
            logging.info(log)
        # ------------------------------------------------------------------
        step += 1

    total_success = []
    total_spl = []
    for acc in episode_success:
        total_success.append(acc)
    for spl in episode_spl:
        total_spl.append(spl)

    if len(total_spl) > 0:
        log = "Average SR/SPL:"
        log += " {:.5f}/{:.5f},".format(
            np.mean(total_success),
            np.mean(total_spl))

    print(log)
    logging.info(log)
        
    total = {'succ': total_success, 'spl': total_spl}

    with open('{}/total.json'.format(
            args.log_dir), 'w') as f:
        json.dump(total, f)


if __name__ == "__main__":
    main()
