import argparse
import subprocess
from pathlib import Path

import numpy as np
import scipy
from skimage.io import imsave, imread
from tqdm import tqdm

from dataset.database import parse_database_name, get_ref_point_cloud
from estimator import name2estimator
from eval import visualize_intermediate_results
from prepare import video2image
from utils.base_utils import load_cfg, project_points
from utils.draw_utils import pts_range_to_bbox_pts, draw_bbox_3d
from utils.pose_utils import pnp


from panda3d.core import DirectionalLight, PointLight, TransparencyAttrib, AmbientLight
from panda3d.core import CollisionTraverser, CollisionHandlerQueue, CollisionHandlerEvent
from panda3d.core import CollisionNode, CollisionBox
from panda3d.core import Point3, LPoint3, LPoint2
from panda3d.core import Lens, LineSegs, NodePath
from panda3d.core import LQuaternionf

from direct.showbase.ShowBase import ShowBase
from direct.showbase import DirectObject
from direct.task import Task

from main import MyApp

import cv2

import random
import uuid
import time
import multiprocessing

RED = (0.9, 0.2, 0.2, 1)
BLUE = (0.2, 0.2, 0.9, 1)
GREEN = (0.2, 0.9, 0.2, 1)

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_normal(points):
    mid = np.mean(points, axis=0)
    u = points[0] - mid
    v = points[1] - mid
    return np.cross(u, v)

def get_normals(points: np.ndarray):
    bools = np.equal(points, np.max(points, axis=0))
    ret = []
    for i in range(3):
        test_points = points[bools[:, i] == True]
        normal = get_normal(test_points)
        ret.append(normal)
    return ret

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


class GameScene(ShowBase):
    def __init__(self, shared_state):
        ShowBase.__init__(self)
        # self.disableMouse()

        self.shared_state = shared_state
        
        self.boxes = []
        self.speed = -0.05
        # self.speed = -0.005
        self.p = 0.01

        self.cube_path = 'models/box'
        
        self.time_sleep = 100 #ms

        self.phi = 30

        self.camLens.setFar(80)

        self.pose = None

        self.magic_translation = np.array([[0, 20, 0]]).T


        self.cam_to_panda = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]
        ]).T

        while not self.shared_state['started']:
            pass
        
        self.setup_mouse_box()
        # self.setup_collision()
        self.make_stick()
        self.setupDefaultLights()

        # self.taskMgr.add(self.generate_box, 'checkBox')
        self.taskMgr.add(self.move_box, 'moveBox')
        self.taskMgr.add(self.plot_mouse, 'plotmouse')
        # self.taskMgr.add(self.check_collision, 'checkCollision')
        self.taskMgr.add(self.move_stick, 'move_stick')

    def pose_to_Rt(self, pose):
        return pose[:,:3], pose[:,3:4] # 3x3, 3x1


    def get_basis(self, coor='camera'):
        if coor == 'panda':
            return np.array([0., 0., 1.])
        elif coor == 'camera':
            return np.array([0., 1., 0.])
        else:
            raise ValueError('unknown coor type')


    def get_top_vec(self, R, coor='camera'):
        return R @ self.get_basis(coor)

    
    def get_calibrated_pose(self, pose: np.array, prev_pose: np.array, bbox):
        if prev_pose is None:
            return pose

        R, t = self.pose_to_Rt(pose)
        prev_R, prev_t = self.pose_to_Rt(prev_pose)
        top_vec = self.get_top_vec(R, 'camera')
        prev_top_vec = self.get_top_vec(prev_R, 'camera')

        angle = np.rad2deg(angle_between(prev_top_vec, top_vec))

        if (90 - self.phi < angle and angle < 90 + self.phi) or \
            (270 - self.phi < angle and angle < 270 + self.phi):
            # wide box
            normals = get_normals(bbox)
    
        elif 90 + self.phi < angle and angle < 270 - self.phi:
            # thin box, error orientation
            top_vec = -top_vec
            pass
        else:
            # thin box, correct orientation
            # do nothing
            pass
        return pose
    

    def try_flip(self, top_vec):
        if top_vec[2] < 0:
            return -top_vec
        return top_vec

    
    def top_vec_to_R(self, top_vec, coor='panda'):
        basis = self.get_basis(coor)
        return rotation_matrix_from_vectors(basis, top_vec)
    

    def move_stick(self, task):
        pose = self.shared_state['pose']
        bbox = self.shared_state['bbox']
        if pose is None:
            return Task.cont

        # pose = self.get_calibrated_pose(pose, self.pose, bbox)
        # self.pose = pose

        pose = self.get_panda_pose(pose)

        R, t = self.pose_to_Rt(pose)
        top_vec = self.get_top_vec(R, 'panda')
        top_vec = self.try_flip(top_vec)
        R = self.top_vec_to_R(top_vec, 'panda')


        # t = self.stick.getPos()

        t = t + self.magic_translation

        self.set_pose(self.stick, R, t)

        time.sleep(self.time_sleep / 1e3)

        return Task.cont
    
    def rotation_to_quat(self, R):
        return scipy.spatial.transform.Rotation.from_matrix(R).as_quat()
    
    def get_panda_pose(self, pose):
        return self.cam_to_panda @ pose
    
    def set_pose(self, target, R, t):
        target.setQuat(LQuaternionf(*self.rotation_to_quat(R)))
        target.setPos(*t)
        
    
    def make_stick(self):
        self.stick = self.loader.loadModel(self.cube_path)
        self.stick.setPos(*self.magic_translation)
        self.stick.setScale(1., 1, 15)
        self.stick.reparentTo(self.render)

    
    def filmTo3d(self, point, depth):
        x, y = point
        near, far = LPoint3(), LPoint3()
        success = self.camLens.extrude(LPoint2(x, y), near, far)
        if not success:
            print('to3d failed')
        neary = near[1]
        fary = far[1]
        
        toFar = fary - depth
        toNear = depth - neary
        total = fary - neary
        x = (far[0] * toNear + near[0] * toFar) / total
        z = (far[2] * toNear + near[2] * toFar) / total
        return x, depth, z


    def setup_mouse_box(self):
        self.mouseBox = self.loader.loadModel(self.cube_path)
        self.mouseBox.setColor((*GREEN[:3], 0.5))
        self.mouseBox.setScale(0.01)
        self.mouseBox.setPos(0, 0, 0)
        self.mouseBox.hide()
        self.mouseBox.reparentTo(self.render)

    
    def setup_collision(self):
        self.traverser = CollisionTraverser()
        self.traverser.setRespectPrevTransform(True)
        self.handler = CollisionHandlerEvent()

        self.handler.addInPattern('%fn-into-%in')

        cNode = CollisionNode('mouseBox')
        cNode.addSolid(CollisionBox(*self.mouseBox.getTightBounds()))
        cBox = self.mouseBox.attachNewNode(cNode)

        self.mouseCBox = cBox
        self.traverser.addCollider(cBox, self.handler)

    
    def setupDefaultLights(self):
        plight = PointLight('plight')
        plight.setColor((*[0.2]*3, 1))
        plnp = self.render.attachNewNode(plight)
        plnp.setPos(1, -2, 1)
        self.render.setLight(plnp)

        alight = AmbientLight('alight')
        alight.setColor((0.2, 0.2, 0.2, 1))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)
    
    def get_uuid(self):
        return str(uuid.uuid1())
    
    def random_choice(self, choices):
        n = len(choices)
        return choices[random.randint(0, n-1)]
    
    
    def generate_box(self, task):
        if random.random() < self.p:
            box, cBox = self.make_box()
            self.boxes.append(box)
            self.traverser.addCollider(cBox, self.handler)
            box.reparentTo(self.render)
        return Task.cont
    
    def check_collision(self, task):
        self.traverser.traverse(self.render)
        # print('check coll', self.traverser.showCollisions(self.render))

        return Task.cont
        
    def move_box(self, task):
        remove_indices = []
        for i, box in enumerate(self.boxes):
            pos = box.getPos()
            pos[1] = box.getY() + task.time * self.speed # y: forward
            box.setFluidPos(pos)
            # print(box.children)
            if box.getY() < 0:
                remove_indices.append(i)

        remove_indices.reverse()
        for index in remove_indices:
            box = self.boxes.pop(index)
            box.removeNode()
            del box
        # if len(self.boxes):

            # for child in filter(lambda x: str(x).endswith('box'), self.boxes[0].children):
            #     print(child, child.getPos(self.render))
        return Task.cont
    
    
    def make_box(self):
        box = self.loader.loadModel(self.cube_path)
        box.setScale(0.1)

        dx = self.random_choice([-1.5, 1.5])
        dz = self.random_choice([-1, 1])
        # dx, dz = 0, 0
        box.setPos(dx, 90, dz)
        box.setColor((*self.random_choice([RED, BLUE])[:3], 0.5))

        cNode = CollisionNode('box')
        cNode.addSolid(CollisionBox(*box.getTightBounds()))
        cBox = box.attachNewNode(cNode)

        return box, cBox


    def plot_mouse(self, task):
        if self.mouseWatcherNode.hasMouse():
            x = self.mouseWatcherNode.getMouseX()
            y = self.mouseWatcherNode.getMouseY()
            x, y, z = self.filmTo3d((x, y), 5)
            self.mouseBox.show()
            self.mouseBox.setFluidPos(x, y, z)
        else:
            self.mouseBox.hide()
            
        return Task.cont


def weighted_pts(pts_list, weight_num=10, std_inv=10):
    weights=np.exp(-(np.arange(weight_num)/std_inv)**2)[::-1] # wn
    pose_num=len(pts_list)
    if pose_num<weight_num:
        weights = weights[-pose_num:]
    else:
        pts_list = pts_list[-weight_num:]
    pts = np.sum(np.asarray(pts_list) * weights[:,None,None],0)/np.sum(weights)
    return pts

def main(shared_state):
    # scene = GameScene(shared_state)
    # scene.run()
    app = MyApp(shared_state).run()
    app.run()

def run_estimator(args, shared_state):
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print('camera is not open')
        exit(-1)

    cfg = load_cfg(args.cfg)
    ref_database = parse_database_name(args.database)
    estimator = name2estimator[cfg['type']](cfg)
    estimator.build(ref_database, split_type='all')

    object_pts = get_ref_point_cloud(ref_database)
    object_bbox_3d = pts_range_to_bbox_pts(np.max(object_pts,0), np.min(object_pts,0))

    
    pose_init = None
    hist_pts = []
    success, img = capture.read()

    while not shared_state['stop'] and success:
        # generate a pseudo K
        h, w, _ = img.shape
        f=np.sqrt(h**2+w**2)
        K = np.asarray([[f,0,w/2],[0,f,h/2],[0,0,1]],np.float32)

        if pose_init is not None:
            estimator.cfg['refine_iter'] = 1 # we only refine one time after initialization
        # pose_pr, inter_results = estimator.predict(img, K, pose_init=pose_init)
        pose_pr, inter_results = estimator.predict(img, K, pose_init=pose_init)
        pose_init = pose_pr

        pts, _ = project_points(object_bbox_3d, pose_pr, K)

        hist_pts.append(pts)
        pts_ = weighted_pts(hist_pts, weight_num=args.num, std_inv=args.std)
        pose_ = pnp(object_bbox_3d, pts_, K)
        pts__, _ = project_points(object_bbox_3d, pose_, K)

        maxs = np.max(pts__, axis=0)
        mins = np.min(pts__, axis=0)
        area = (maxs[0] - mins[0]) * (maxs[1] - mins[1])

        
        bbox_img_ = draw_bbox_3d(img, pts__, (0,0,255))

        # imsave(f'{str(output_dir)}/images_out_smooth/{que_id}-bbox.jpg', bbox_img_)
        pose_init = pose_
        shared_state['pose'] = pose_
        shared_state['bbox'] = object_bbox_3d
        shared_state['started'] = True

        cv2.imshow('test', bbox_img_)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        success, img = capture.read()



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/gen6d_pretrain.yaml')
    parser.add_argument('--database', type=str, default="custom/mouse")
    parser.add_argument('--output', type=str, default="data/custom/mouse/test")

    # input video process
    parser.add_argument('--video', type=str, default="data/custom/video/mouse-test.mp4")
    parser.add_argument('--resolution', type=int, default=960)
    parser.add_argument('--transpose', action='store_true', dest='transpose', default=False)

    # smooth poses
    parser.add_argument('--num', type=int, default=1)
    parser.add_argument('--std', type=float, default=2.5)

    parser.add_argument('--ffmpeg', type=str, default='ffmpeg')
    args = parser.parse_args()
    # main(args)

    
    with multiprocessing.Manager() as manager:
        shared_state = manager.dict()
        shared_state['started'] = False
        shared_state['stop'] = False
        shared_state['pose'] = None
    
        # shared_state['stop'] = True
        # shared_state['started'] =True
        p = multiprocessing.Process(target=run_estimator, args=(args, shared_state))
        p.start()
        main(shared_state)

        shared_state['stop'] = True
        p.join()