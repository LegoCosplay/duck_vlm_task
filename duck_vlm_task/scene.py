import os
import time
import numpy as np
import mujoco
import mujoco.viewer
base_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.join(base_path, "../microduck_rl/scripts"))
from infer_policy import DEFAULT_POSE, load_bam_model, PolicyInference, BAM_VIN_MIN, BAM_STIFF_SOLREF_FRICTION, BAM_STIFF_SOLIMP_FRICTION, BAM_MODEL

# MICRODUCK_XML = "src/mjlab_microduck/robot/microduck/scene.xml"
MICRODUCK_BALL_XML = os.path.join(base_path, "../microduck_rl/src/mjlab_microduck/robot/microduck/scene_ball.xml")


def load_mujoco_with_bam_from_spec(spec, bam_model, timestep: float, vin_drop_gain, vin_min):
    "公式の関数から、シーンを書き換えてspecが受け取れるように変更したもの"
    from bam.mujoco import MujocoController

    kt = bam_model.kt.value
    R = bam_model.R.value
    force_limit = bam_model.actuator.vin * kt / R

    # spec = mujoco.MjSpec.from_file(xml_path)
    names = []
    for act in spec.actuators:
        tgt = act.target
        tgt_name = tgt.name if hasattr(tgt, "name") else str(tgt)
        if tgt_name.startswith("passive_"):
            continue
        act.set_to_motor()
        act.forcelimited = True
        act.forcerange = (-force_limit, force_limit)
        act.ctrllimited = False
        act.gear = [1.0, 0, 0, 0, 0, 0]
        names.append(act.name)
        for joint in spec.joints:
            if joint.name == tgt_name:
                joint.damping = np.zeros((3, 1))  # MjsJoint expects a (3,1) array
                joint.frictionloss = 0.0
                joint.solref_friction = BAM_STIFF_SOLREF_FRICTION
                joint.solimp_friction = BAM_STIFF_SOLIMP_FRICTION
                break

    model = spec.compile()
    model.opt.timestep = timestep
    data = mujoco.MjData(model)
    bam_ctrl = MujocoController(bam_model, names, model, data,
                                vin_drop_gain=vin_drop_gain, vin_min=vin_min)
    print(f"BAM {BAM_MODEL} actuators on {len(names)} joints: kt={kt:.4f} R={R:.4f} "
          f"vin={bam_model.actuator.vin:.2f}V kp_fw={bam_model.actuator.kp:.0f} "
          f"vin_drop_gain={vin_drop_gain} vin_min={vin_min} "
          f"max_current={bam_model.actuator.max_current} forcerange=+/-{force_limit:.3f}Nm "
          f"armature={bam_model.actuator.get_extra_inertia():.2e}")
    return model, data, bam_ctrl, names


def main():
    # 初期化（spec読み込みとボールの追加）
    xml_path = MICRODUCK_BALL_XML
    walking_onnx_path = os.path.join(base_path, "policies/alpha_walking.onnx")
    standing_onnx_path = os.path.join(base_path, "policies/alpha_stand.onnx")
    kick_left_onnx_path = os.path.join(base_path, "policies/ball_kick_left.onnx")
    kick_right_onnx_path = os.path.join(base_path, "policies/ball_kick_right.onnx")
    decimation = 4
    BALL_OFFSET_X = 0.09
    BALL_OFFSET_ABS_Y = 0.042
    BALL_RADIUS = 0.035

    print(f"Loading MuJoCo model from: {xml_path}")
    spec = mujoco.MjSpec.from_file(xml_path)
    # model = mujoco.MjModel.from_xml_path(xml_path) # こちらだとコンパイル後で編集ができない

    # プロパティについては、https://mujoco.readthedocs.io/en/stable/python.html#に書いてある。
    spec.modelname = "my_ball_scene"
    body = spec.worldbody.add_body(
        name="my_blue_sphere",
        pos=[0.3, -0.5, 0.035],
        ipos=[0, 0, 0], # ボールの重心位置はボールの中心
        inertia=[1.225e-5, 1.225e-5, 1.225e-5],
        mass=0.015,
        explicitinertial=True, # Trueにしておけば、inertialの値を算出ではなく、指定した値が使える
    )
    geom = body.add_geom(
        name='my_sphere',
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.035, 0, 0],
        rgba=[0, 0, 1, 1],
        # mass=0.015,
        friction=[0.5, 0.005, 0.0001],
    )

    body.add_freejoint() # freejointはデフォルトでtype=mjJNT_FREEなので省略可能
    # model = spec.compile()
    # data = mujoco.MjData(model)
    # print("Legacy MuJoCo position actuators (--no-bam): NOT the actuator the policy was trained with")

    # BAMモデルの値はmicroduckファームウェアの参考値から
    kp_fw = 200.0
    vin = 7.4
    current_limit = 0.0
    vin_drop_gain = 0.1

    bam_model = load_bam_model(kp_fw, vin, current_limit)
    model, data, bam_ctrl, _bam_names = load_mujoco_with_bam_from_spec(
        spec, bam_model, 0.005, vin_drop_gain, BAM_VIN_MIN)
    # mujoco.viewer.launch(model, data)
    control_dt = decimation * model.opt.timestep # 実時間でやると転ぶ時と、転ばないときがあるので、sim時間で動かす

    # Initialize policy
    policy = PolicyInference(model, data, bam_ctrl=bam_ctrl, new_cmd_obs=True, use_projected_gravity=True, walking_onnx_path=walking_onnx_path, standing_onnx_path=standing_onnx_path, kick_left_onnx_path=kick_left_onnx_path, kick_right_onnx_path=kick_right_onnx_path)
    policy.set_vel_cmd(0,0,0)

    # ローラーブレードじゃないときは以下の値
    policy.vel_max_x = 0.3
    policy.vel_min_x = -0.3
    policy.vel_max_y = 0.2
    policy.vel_min_y = -0.2
    policy.vel_max_ang = 1.5

    # 初期姿勢
    freejoint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qpos_adr = model.jnt_qposadr[freejoint_id]
    data.qpos[qpos_adr + 0] = 0.0
    data.qpos[qpos_adr + 1] = 0.0
    data.qpos[qpos_adr + 2] = 0.125  # rollers add 13.5mm height
    data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]
    for i, qpos_idx in enumerate(policy.joint_qpos_indices): # 各関節を初期姿勢にする
        data.qpos[qpos_idx] = policy.default_pose[i]
    bam_ctrl.reset(data.qpos)   # clears voltage-drop state, q_target = current qpos
    policy.set_position_targets(policy.default_pose)
    mujoco.mj_forward(model, data)

    # phase 0: [wait] 立って待つ→1秒経過後前進指令
    # phase 1: [walk] 前進→n秒経過後停止指令
    # phase 2: [stop] 停止して待機→1秒経過後キック
    # phase 3: [kick] キック→キックが終わったら待機

    # loopで使うためにはlaunch_passiveを使う（自分の処理を挟むにはこちらでやる必要あり）
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        start_time = time.time()
        prev_step_time = start_time
        kicked = False
        phase = "wait"

        while viewer.is_running():# and not quit_requested:
            step_start = time.time()

            actual_dt = step_start - prev_step_time
            prev_step_time = step_start
            policy.update_behavior(control_dt)

            action = policy.infer()
            policy.apply_action(action)

            for _ in range(decimation):
                bam_ctrl.update()
                mujoco.mj_step(model, data)

            viewer.sync()
            elapsed = time.time() - start_time
            if phase=="wait" and data.time > 1.0:
                policy.set_vel_cmd(0.3, 0, 0) # 前進
                phase = "walk"
            elif phase=="walk" and data.time > 2.5:
                policy.set_vel_cmd(0, 0, 0) # 停止
                phase = "stop"
            elif phase=="stop" and data.time > 4.0:
                policy.trigger_behavior("kick_left")
                phase = "kick"


if __name__ == "__main__":
    main()
