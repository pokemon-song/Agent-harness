"""Export every simulation-time interval with visible A/B labels; no physics/API calls."""
import argparse
import json
import math
from pathlib import Path
import re
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from go2_demo.occluded_video import _ffmpeg, _sha


def project(points, camera, width, height):
    points = np.atleast_2d(points)
    forward = np.asarray(camera.forward, dtype=float)
    up = np.asarray(camera.up, dtype=float)
    right = np.cross(forward, up)
    relative = points - camera.pos
    depth = relative @ forward
    near = camera.frustum_near
    vertical = camera.frustum_top - camera.frustum_bottom
    horizontal = camera.frustum_width or vertical * width / height
    left = camera.frustum_center - horizontal / 2
    u = width * ((relative @ right) * near / depth - left) / horizontal
    v = height * (1 - ((relative @ up) * near / depth - camera.frustum_bottom) / vertical)
    return np.column_stack((u, v)), depth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preview', type=Path)
    args = parser.parse_args()
    if args.output.exists() and args.preview is None:
        raise FileExistsError(args.output)
    video_dir = args.arm / 'video'
    capture = json.loads((video_dir / 'capture.json').read_text())
    metrics = json.loads((args.arm / 'metrics.json').read_text())
    episode_manifest = json.loads((args.arm / 'manifest.json').read_text())
    mode = episode_manifest['mode']
    assert metrics['mode'] == mode
    memory_label = {'empty': '无外部记忆', 'query': '按需记忆', 'ledger': '轨迹记忆'}[mode]
    navigation = episode_manifest['navigation']
    navigation_label = {'reactive_no_subgoals_v1': '无显式子目标',
                        'persistent_subgoals_v1': '有显式子目标'}[navigation]
    display_title = f'{navigation_label} · {memory_label}'
    footer_text = '蓝线：A 阶段   绿线：B 阶段   |   全部仿真时间原速保留，无加速删减   |   A/B 标记仅用于回放'
    if metrics.get('score_policy_version') == 'b_distance_tiers_v1':
        points, score = metrics['checkpoint_points'], metrics['b_distance_score']
        distance, error = score['distance_to_target_center_m'], score['absolute_error_m']
        distance_text = f'B 距离 {distance:.3f} m / 误差 {error:.3f} m' if distance is not None else 'B 距离无有效测量'
        status_text = '通过' if metrics['success'] else '未通过'
        footer_text = (f"本轮最终：A {points['A']} + B {points['B']} − 碰撞 {metrics['collision_penalty']} = {metrics['score']}/100"
                       f'   |   {distance_text}   |   严格完成：{status_text}（回放评分）')
    assert _sha(video_dir / 'model.mjb') == capture['model_sha256']
    assert _sha(video_dir / 'poses.npz') == capture['poses_sha256']
    assert capture['mujoco_version'] == mujoco.__version__
    with np.load(video_dir / 'poses.npz', allow_pickle=False) as data:
        times, poses = data['time'], data['qpos']
    assert len(times) == capture['sample_count'] and np.isfinite(times).all() and np.isfinite(poses).all()
    assert np.all(np.diff(times) > 0) and times[0] == capture['start_s'] and times[-1] == capture['end_s']
    assert abs(times[-1] - metrics['final_sim_s']) < 1e-8
    fps = 12.5
    full_count = max(2, math.ceil((times[-1]-times[0])*fps-1e-8)+1)
    selected = np.arange(full_count,dtype=int)
    ticks = times[0] + np.arange(full_count) / fps
    source = np.searchsorted(times, ticks + 1e-9, side='right') - 1
    source[0], source[-1] = 0, len(times) - 1
    source = source[selected]
    rows = [json.loads(line) for line in (args.arm / 'trajectory.jsonl').read_text().splitlines()]
    assert len(rows) + 1 == len(times)

    width, view_height, header, footer = 1280, 864, 64, 32
    height = header + view_height + footer
    model = mujoco.MjModel.from_binary_path(str(video_dir / 'model.mjb'))
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = view_height
    model.vis.quality.offsamples = 0
    state = mujoco.MjData(model)
    state.qpos[:] = poses[0]
    mujoco.mj_forward(model, state)
    robot = np.vstack([state.body('base').xpos.copy(), [row['base_xyz_m'] for row in rows]])
    targets = {name: state.geom_xpos[model.geom(geom).id].copy()
               for name, geom in (('A', 'scenario_target'), ('B', 'memory_goal_b'))}
    bounds = np.vstack([robot[:, :2], *[point[:2] for point in targets.values()]])
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:2] = (bounds.min(axis=0) + bounds.max(axis=0)) / 2
    camera.lookat[2] = .2
    camera.azimuth, camera.elevation = 90, -82
    camera.distance = 14.5
    model.vis.map.zfar = max(model.vis.map.zfar,
                            (camera.distance + np.linalg.norm(np.ptp(bounds, axis=0)) + 5) / model.stat.extent)
    regular = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    bold = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
    font = ImageFont.truetype(regular, 22)
    small = ImageFont.truetype(regular, 18)
    title = ImageFont.truetype(bold, 27)
    target_font = ImageFont.truetype(bold, 28)
    blue, green = (28, 87, 218), (13, 133, 62)
    handoff = capture['handoff_s']
    handoff_index = int(np.searchsorted(times,handoff)) if handoff is not None else len(times)-1
    if handoff is not None:
        assert handoff_index < len(times) and times[handoff_index] == handoff

    with mujoco.Renderer(model, width=width, height=view_height) as renderer:
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        footprint = np.vstack([robot+[dx,dy,dz] for dx in (-.65,.65) for dy in (-.55,.55) for dz in (-.3,.4)])
        for _ in range(30):
            model.vis.map.zfar = max(model.vis.map.zfar,
                (camera.distance + np.linalg.norm(np.ptp(bounds,axis=0)) + 5)/model.stat.extent)
            renderer.update_scene(state,camera=camera)
            gl_camera = mujoco.mjv_averageCamera(renderer.scene.camera[0],renderer.scene.camera[1])
            footprint_pixels,depths = project(footprint,gl_camera,width,view_height)
            target_pixels,target_depths = project(np.array(list(targets.values())),gl_camera,width,view_height)
            if (footprint_pixels[:,0].min()>=12 and footprint_pixels[:,0].max()<width-12
                and footprint_pixels[:,1].min()>=12 and footprint_pixels[:,1].max()<view_height-12
                and depths.min()>gl_camera.frustum_near and depths.max()<gl_camera.frustum_far
                and target_pixels[:,0].min()>45 and target_pixels[:,0].max()<width-180
                and target_pixels[:,1].min()>75 and target_pixels[:,1].max()<view_height-45
                and target_depths.min()>gl_camera.frustum_near and target_depths.max()<gl_camera.frustum_far):
                break
            camera.distance *= 1.1
        else:
            raise ValueError('Could not fit both goals and the full robot trajectory')
        screen, depths = project(robot, gl_camera, width, view_height)
        target_screen = {label: project(point, gl_camera, width, view_height)[0][0]
                         for label, point in targets.items()}
        # Check the entire recorded route with body clearance, not just its endpoints.
        projected, clip_depths = project(footprint, gl_camera, width, view_height)
        assert projected[:, 0].min() >= 8 and projected[:, 0].max() < width - 8
        assert projected[:, 1].min() >= 8 and projected[:, 1].max() < view_height - 8
        assert clip_depths.min() > gl_camera.frustum_near and clip_depths.max() < gl_camera.frustum_far
        for label, point in targets.items():
            pixel, depth = project(point, gl_camera, width, view_height)
            assert 45 < pixel[0, 0] < width - 180 and 75 < pixel[0, 1] < view_height - 45
            assert gl_camera.frustum_near < depth[0] < gl_camera.frustum_far
        screen[:, 1] += header
        for point in target_screen.values():
            point[1] += header

        def frame(index):
            state.qpos[:] = poses[index]
            state.time = times[index]
            mujoco.mj_forward(model, state)
            renderer.update_scene(state, camera=camera)
            image = Image.new('RGB', (width, height), (18, 27, 42))
            image.paste(Image.fromarray(renderer.render()), (0, header))
            draw = ImageDraw.Draw(image)
            split = min(index, handoff_index)
            trace_a = screen[np.r_[np.arange(0, split, 16), split]]
            if len(trace_a) > 1:
                draw.line([tuple(point) for point in trace_a], fill=(91, 147, 244), width=3)
            if handoff is not None and index > handoff_index:
                trace_b = screen[np.r_[np.arange(handoff_index, index, 16), index]]
                draw.line([tuple(point) for point in trace_b], fill=(40, 185, 108), width=3)
            for label, color, name in (('A', blue, 'A  蓝球'), ('B', green, 'B  绿方块')):
                x, y = target_screen[label]
                radius = 22
                draw.ellipse((x-radius-2, y-radius-2, x+radius+2, y+radius+2), outline='white', width=5)
                draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=color, width=3)
                tx, ty = x+42, y-64
                box = draw.textbbox((0, 0), name, font=target_font)
                tw = box[2] - box[0]
                draw.line((x+17, y-17, tx, ty+24), fill='white', width=6)
                draw.line((x+17, y-17, tx, ty+24), fill=color, width=3)
                draw.rounded_rectangle((tx-10, ty-3, tx+tw+13, ty+43), radius=9,
                                       fill=color, outline='white', width=2)
                draw.text((tx, ty), name, font=target_font, fill='white')
            x, y = screen[index]
            draw.rounded_rectangle((x-32, y+29, x+40, y+57), radius=5, fill=(34, 42, 52))
            draw.text((x-25, y+29), '机器人', font=small, fill=(255, 226, 94))
            returning = handoff is not None and times[index] >= handoff
            stage = '返程：返回 B' if returning else ('A 阶段（无返程）' if handoff is None else '去程：寻找 A')
            draw.text((18, 8), display_title, font=title, fill='white')
            draw.text((520, 13), stage, font=font, fill=(120, 236, 172) if returning else (149, 191, 255))
            draw.text((936, 14), f'仿真时间 {times[index]:.2f} s', font=font, fill='white')
            draw.text((18, height-29), footer_text, font=small, fill=(218, 228, 240))
            return image

        camera_info = {'lookat': camera.lookat.tolist(), 'distance': camera.distance,
                       'azimuth': camera.azimuth, 'elevation': camera.elevation,
                       'near_clip_m': float(gl_camera.frustum_near), 'far_clip_m': float(gl_camera.frustum_far),
                       'target_pixels': {key: point.tolist() for key, point in target_screen.items()},
                       'whole_robot_trajectory_in_frame': True, 'both_targets_and_labels_in_frame': True}
        if args.preview is not None:
            previews = [('start',0),('final',len(times)-1)]
            if handoff is not None:
                previews.insert(1,('handoff',handoff_index))
            args.preview.parent.mkdir(parents=True,exist_ok=True)
            for suffix, index in previews:
                file = args.preview.with_name(args.preview.stem + '-' + suffix + '.png')
                frame(index).save(file)
                print(file, flush=True)
            print(json.dumps(camera_info), flush=True)
            return

        binary = _ffmpeg()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        partial = args.output.with_suffix('.partial.mp4')
        process = subprocess.Popen([binary, '-hide_banner', '-loglevel', 'error', '-y',
                                    '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
                                    '-framerate', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
                                    '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p',
                                    '-movflags', '+faststart', str(partial)], stdin=subprocess.PIPE)
        try:
            for number, index in enumerate(source):
                process.stdin.write(frame(index).tobytes())
                if number % 500 == 0:
                    print(f'rendered {number}/{len(source)}', flush=True)
        finally:
            process.stdin.close()
            status = process.wait()
        if status:
            raise RuntimeError(f'ffmpeg exited {status}')
    verified = subprocess.run([binary, '-hide_banner', '-loglevel', 'error', '-xerror', '-i', str(partial),
                               '-map', '0:v:0', '-f', 'null', '-', '-progress', 'pipe:1', '-nostats'],
                              check=True, capture_output=True, text=True)
    decoded = re.findall(r'^frame=(\d+)$', verified.stdout, re.MULTILINE)
    assert decoded and int(decoded[-1]) == len(source)
    partial.replace(args.output)
    manifest = {'file': args.output.name, 'sha256': _sha(args.output), 'frame_count': len(source),
                'fps': fps, 'duration_s': len(source)/fps, 'width': width, 'height': height,
                'source_capture': capture, 'selected_source_frames': selected.tolist(),
                'initial_and_final_states_included': True, 'decoded_all_frames': True,
                'physics_or_api_calls': False, 'annotations_are_observer_only': True,
                'camera': camera_info, 'episode_id': metrics['episode_id'],
                'mode': mode, 'navigation': navigation, 'display_title': display_title,
                'footer_text': footer_text, 'score_policy_version': metrics.get('score_policy_version'),
                'entire_simulation_time_included': True, 'simulation_duration_s': float(times[-1]-times[0]),
                'return_phase_exists': handoff is not None,
                'edit': 'none: original simulation time, all output ticks retained',
                'source_exporter_sha256': _sha(Path(__file__))}
    args.output.with_suffix('.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'output': str(args.output), 'duration_s': manifest['duration_s'],
                      'frame_count': len(source)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
