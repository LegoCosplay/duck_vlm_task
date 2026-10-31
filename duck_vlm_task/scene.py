import os
import mujoco
import mujoco.viewer


base_path = os.path.dirname(os.path.abspath(__file__))
# MICRODUCK_XML = "src/mjlab_microduck/robot/microduck/scene.xml"
MICRODUCK_BALL_XML = os.path.join(base_path, "../microduck_rl/src/mjlab_microduck/robot/microduck/scene_ball.xml")

xml_path = MICRODUCK_BALL_XML

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

# model.opt.timestep = 0.005
model = spec.compile()
data = mujoco.MjData(model)
print("Legacy MuJoCo position actuators (--no-bam): NOT the actuator the policy was trained with")
mujoco.viewer.launch(model, data)

# BAMモデルの値はmicroduckファームウェアの参考値から
kp_fw = 200.0
vin = 7.4
current_limit = 0.0

bam_model = load_bam_model(kp_fw, vin, current_limit)
vin_drop_gain = args.vin_drop_gain if args.vin_drop_gain > 0 else None
model, data, bam_ctrl, _bam_names = load_mujoco_with_bam(
    xml_path, bam_model, 0.005, vin_drop_gain, BAM_VIN_MIN)