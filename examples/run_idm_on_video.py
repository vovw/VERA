"""Run the VERA inverse-dynamics model (IDM) on your OWN video.

VERA is two stages: a video *planner* that "dreams" future frames, and an *IDM*
that translates a video into robot actions. This script exercises ONLY the IDM:
you hand it a real video (your own footage, a teleop replay, anything) instead of
a dream, and it recovers the per-step action that best explains the motion between
consecutive frames.

Pipeline (per consecutive frame pair t -> t+1, exactly what MotionPolicy does):
  1. IDM.compute_jacobian(frame_t)         -> J : per-pixel map (action -> pixel flow)
  2. tracker(frame_t, frame_t+1)           -> y : the observed pixel motion in YOUR video
  3. controller.solve(J, y)                -> du: the action that produced that motion

Usage:
  python examples/run_idm_on_video.py --video path/to/clip.mp4 --out actions.npy
  python examples/run_idm_on_video.py --video path/to/frames_dir --out actions.npy
  python examples/run_idm_on_video.py --video path/to/frames.npy --out actions.npy   # [T,H,W,3]

Checkpoints default to ./vera-ckpts (the Wave-1 PushT IDM). Override with
--dynamics-ckpt / --planner-ckpt or the VERA_PUSHT_* env vars.

NOTE on domain: the IDM is embodiment-specific. The PushT IDM recovers 2D planar
pusher actions and only produces meaningful numbers on PushT-like top-down footage.
Point --dynamics-ckpt at a different IDM to translate a different embodiment.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

# Default to the downloaded Wave-1 PushT checkpoints unless the env says otherwise.
_REPO = Path(__file__).resolve().parent.parent
_DEF_DYN = _REPO / "vera-ckpts" / "pusht-idm" / "model.ckpt"
_DEF_PLAN = _REPO / "vera-ckpts" / "pusht-dfot" / "model.ckpt"


def load_video(path: str) -> torch.Tensor:
    """Return frames as float tensor [1, T, 3, H, W] in [0, 1]."""
    p = Path(path)
    if p.is_dir():
        import imageio.v3 as iio

        files = sorted(
            f for f in p.iterdir()
            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
        if not files:
            raise FileNotFoundError(f"no image frames found in {p}")
        frames = np.stack([iio.imread(f) for f in files])  # [T,H,W,3] uint8
    elif p.suffix.lower() == ".npy":
        frames = np.load(p)  # [T,H,W,3]
    else:
        import imageio.v3 as iio

        frames = iio.imread(p)  # mp4 -> [T,H,W,3]
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected [T,H,W,3] frames, got {frames.shape}")
    t = torch.from_numpy(frames).float()
    if float(t.max()) > 1.5:  # uint8-ish -> [0,1]
        t = t / 255.0
    return t.permute(0, 3, 1, 2).unsqueeze(0).contiguous()  # [1,T,3,H,W]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the VERA IDM on a custom video")
    ap.add_argument("--video", required=True, help="mp4 / frames-dir / .npy [T,H,W,3]")
    ap.add_argument("--out", default="actions.npy", help="output .npy for per-step actions")
    ap.add_argument("--dynamics-ckpt", default=os.environ.get("VERA_PUSHT_DYNAMICS_CKPT", str(_DEF_DYN)))
    ap.add_argument("--planner-ckpt", default=os.environ.get("VERA_PUSHT_PLANNER_CKPT", str(_DEF_PLAN)))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--denormalize", action="store_true",
                    help="map solved du back to physical action units (else raw model du)")
    args = ap.parse_args()

    device = torch.device(args.device)

    # The IDM lives inside MotionPolicy (alongside the controller). We reuse the
    # PushT factory so checkpoint loading + normalization matches the server exactly.
    os.environ.setdefault("VERA_PUSHT_DYNAMICS_CKPT", args.dynamics_ckpt)
    os.environ.setdefault("VERA_PUSHT_PLANNER_CKPT", args.planner_ckpt)
    from vera.server.start_server_pusht import build_policy
    from vera.policy.base_policy import PolicyObservation
    from vera.policy.world_models.alltracker_inference import AllTrackerInference
    from vera.policy.world_models.runtime_motion_tracks import RuntimeMotionTracks

    print(f"[idm] building policy (idm={args.dynamics_ckpt})")
    policy = build_policy(device, planner_ckpt=args.planner_ckpt, dynamics_ckpt=args.dynamics_ckpt)
    policy.reset()

    video = load_video(args.video).to(device)
    B, T, C, H, W = video.shape
    if T < 2:
        raise ValueError("need at least 2 frames")
    print(f"[idm] video: {T} frames @ {H}x{W}")

    # Observed pixel motion between every consecutive pair (source = frame t).
    # We track each pair independently so disp[t] is the t -> t+1 motion with the
    # source anchored at frame t -- the convention _track_rgb_chunk_to_actions wants.
    tracker = AllTrackerInference(device=device)
    pair_tracks = [
        tracker.infer(video[:, t:t + 2], return_visualization=False).motion_tracks
        for t in range(T - 1)
    ]
    tracks = RuntimeMotionTracks(
        xy_src=torch.cat([p.xy_src for p in pair_tracks], dim=1),
        xy_tgt=torch.cat([p.xy_tgt for p in pair_tracks], dim=1),
        vis_src=torch.cat([p.vis_src for p in pair_tracks], dim=1),
        vis_tgt=torch.cat([p.vis_tgt for p in pair_tracks], dim=1),
        idx_src=torch.cat([p.idx_src for p in pair_tracks], dim=1),
        idx_tgt=torch.cat([p.idx_tgt for p in pair_tracks], dim=1),
        image_size=pair_tracks[0].image_size,
    ).as_policy_dict()

    obs = PolicyObservation(
        rgb=video[:, 0].permute(0, 2, 3, 1).cpu().numpy(),  # [B,H,W,3]
        q_robot=None,  # unused by the IDM flow/track path
    )
    source_rgb = video[:, : T - 1]
    target_rgb = video[:, 1:T]
    widths = policy._source_view_widths_for_obs(obs, W)

    with torch.no_grad():
        actions, _vis = policy._track_rgb_chunk_to_actions(
            obs, source_rgb, tracks, widths, target_rgb=target_rgb,
        )
        if args.denormalize:
            actions = policy.denormalize_dynamics_action(actions)

    out = actions[0].detach().cpu().numpy()  # [T-1, action_dim]
    np.save(args.out, out)
    print(f"[idm] solved {out.shape[0]} actions, dim={out.shape[1]} -> {args.out}")
    print(f"[idm] per-step action norm (mean): {np.linalg.norm(out, axis=-1).mean():.4f}")


if __name__ == "__main__":
    main()
