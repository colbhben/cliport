"""Put-block-in-bowl, human-in-the-loop variant.

Differences from PutBlockInBowlSeenColors:
- 4-color pool: blue, red, green, yellow (HITL_COLORS).
- Exactly one target block and one target bowl per episode.
- All distractors are drawn from the same 4-color pool, so the human cannot
  identify the target by elimination.
- Exposes per-episode metadata on `self`:
    block_color, bowl_color (color names)
    target_block_id, target_bowl_id
    block_ids, bowl_ids (all ids spawned, including distractors and target)
- The task `goals[0]` matrix is intentionally permissive: any of the spawned
  blocks may match any of the spawned bowls. The recorder rewrites
  `self.goals` after each click to constrain the oracle to the clicked
  object.

Recorder usage:
    task = PutBlockInBowlHITL()
    task.allowed_pairs = [(b,w) for b in HITL_COLORS for w in HITL_COLORS if b != w]
    env.set_task(task)
    obs = env.reset()
    # task.target_block_id, task.target_bowl_id, task.block_color etc. set
"""

import numpy as np
import pybullet as p
import random

from cliport.tasks.task import Task
from cliport.utils import utils


HITL_COLORS = ['blue', 'red', 'green', 'yellow']


class PutBlockInBowlHITL(Task):
    """4-color HITL put-block-in-bowl with exactly one target of each kind."""

    def __init__(self):
        super().__init__()
        self.max_steps = 3
        self.pos_eps = 0.05
        self.lang_template = "put the {pick} block in the {place} bowl"
        self.task_completed_desc = "done placing block in bowl."

        # Recorder may overwrite `allowed_pairs` between resets to constrain
        # color sampling to a particular split (train / val_unseen).
        self.allowed_pairs = [(b, w) for b in HITL_COLORS for w in HITL_COLORS if b != w]

        # Per-episode metadata, populated by reset().
        self.block_color = None
        self.bowl_color = None
        self.target_block_id = None
        self.target_bowl_id = None
        self.block_ids = []
        self.bowl_ids = []

    def get_colors(self):
        return list(HITL_COLORS)

    def reset(self, env):
        super().reset(env)

        # Sample a (block_color, bowl_color) tuple from the allowed pool.
        block_color, bowl_color = random.choice(self.allowed_pairs)
        self.block_color = block_color
        self.bowl_color = bowl_color

        block_rgb = utils.COLORS[block_color]
        bowl_rgb = utils.COLORS[bowl_color]

        # ---- Spawn one bowl per HITL color (target + 3 distractors). ----
        bowl_size = (0.12, 0.12, 0)
        bowl_urdf = 'bowl/bowl.urdf'
        bowl_ids = []
        bowl_poses = []
        bowl_id_to_color = {}
        target_bowl_id = None
        # Shuffle so position doesn't leak color identity.
        bowl_color_order = list(HITL_COLORS)
        random.shuffle(bowl_color_order)
        for cn in bowl_color_order:
            pose = self.get_random_pose(env, bowl_size)
            if not pose:
                continue
            bowl_id = env.add_object(bowl_urdf, pose, 'fixed')
            if bowl_id is None:
                continue
            p.changeVisualShape(bowl_id, -1, rgbaColor=utils.COLORS[cn] + [1])
            bowl_ids.append(bowl_id)
            bowl_poses.append(pose)
            bowl_id_to_color[bowl_id] = cn
            if cn == bowl_color:
                target_bowl_id = bowl_id

        # ---- Spawn one block per HITL color (target + 3 distractors). ----
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        block_ids = []
        block_id_to_color = {}
        target_block_id = None
        block_color_order = list(HITL_COLORS)
        random.shuffle(block_color_order)
        block_objs = []
        for cn in block_color_order:
            pose = self.get_random_pose(env, block_size)
            if not pose:
                continue
            block_id = env.add_object(block_urdf, pose)
            if block_id is None:
                continue
            p.changeVisualShape(block_id, -1, rgbaColor=utils.COLORS[cn] + [1])
            block_ids.append(block_id)
            block_id_to_color[block_id] = cn
            block_objs.append((block_id, (0, None)))
            if cn == block_color:
                target_block_id = block_id

        if target_block_id is None or target_bowl_id is None:
            # Re-attempt this episode if a urdf placement failed.
            raise RuntimeError(
                f"PutBlockInBowlHITL.reset failed: "
                f"target_block={target_block_id}, target_bowl={target_bowl_id}"
            )

        # Permissive goal matrix; recorder narrows it after the click.
        match_matrix = np.ones((len(block_objs), len(bowl_poses)), dtype=np.int32)
        self.goals.append((
            block_objs, match_matrix, bowl_poses,
            False,  # replace
            True,   # rotations
            'pose', None, 1,
        ))
        self.lang_goals.append(self.lang_template.format(
            pick=block_color, place=bowl_color))

        # Stash metadata for the recorder.
        self.block_ids = block_ids
        self.bowl_ids = bowl_ids
        self.target_block_id = target_block_id
        self.target_bowl_id = target_bowl_id
        self.block_id_to_color = block_id_to_color
        self.bowl_id_to_color = bowl_id_to_color
        self.bowl_poses = bowl_poses
        self.block_objs = block_objs

    # ---------------------------------------------------------------------
    # Helpers used by the recorder to drive the oracle for the *clicked*
    # block / bowl rather than the goal-derived match.
    # ---------------------------------------------------------------------

    def constrain_goal_to_click(self, clicked_block_id, clicked_bowl_id):
        """Rewrite goals[0] so the oracle picks exactly the clicked block
        and places it in exactly the clicked bowl.
        """
        block_objs = [(clicked_block_id, (0, None))]
        bowl_idx = self.bowl_ids.index(clicked_bowl_id)
        targs = [self.bowl_poses[bowl_idx]]
        match = np.ones((1, 1), dtype=np.int32)
        # Preserve the rest of the original goal tuple.
        _, _, _, replace, rotations, metric, params, max_reward = self.goals[0]
        self.goals[0] = (block_objs, match, targs, replace, rotations,
                         metric, params, max_reward)
