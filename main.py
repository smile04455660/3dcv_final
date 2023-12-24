from math import pi, sin, cos

from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import globalClock
from direct.task import Task
from direct.actor.Actor import Actor
from direct.interval.IntervalGlobal import Sequence
from panda3d.core import Geom,GeomVertexFormat,GeomVertexData,GeomTriangles,GeomVertexWriter, GeomNode, OrthographicLens, load_prc_file, Filename
from panda3d.core import TextNode, CollisionNode, CollisionBox, CollisionHandlerEvent, Point3,CollisionTraverser, Point2, Vec3,TextureStage
from panda3d.core import PointLight, AmbientLight, NodePath
from panda3d.core import LQuaternionf
import random
import os
import numpy as np
import scipy

load_prc_file('myConfig.prc')


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


def create_rec(x,z,width,height,color=None):
    _format = GeomVertexFormat.getV3c4()
    vdata = GeomVertexData('square',_format,Geom.UHDynamic)
    vertex = GeomVertexWriter(vdata,'vertex')
    color = GeomVertexWriter(vdata,'color')
    vertex.addData3(x,0,z)
    vertex.addData3(x + width,0,z)
    vertex.addData3(x + width,0,z + height)
    vertex.addData3(x,0,z + height)
    
    color.addData4f(1,0,0,1)
    color.addData4f(1,0,0,1)
    color.addData4f(1,0,0,1)
    color.addData4f(1,0,0,1)
    
    tris = GeomTriangles(Geom.UHDynamic)
    tris.addVertices(0,1,2)
    tris.addVertices(2,3,0)
    
    square = Geom(vdata)
    square.addPrimitive(tris)
    return square
    
    

class MyApp(ShowBase):
    def __init__(self, shared_state):
        ShowBase.__init__(self)

        self.shared_state = shared_state
        
        self.set_background_color(0,0,0,0)
        
        
        self.accept('arrow_left',self.generate_box,['left'])
        self.accept('arrow_right',self.generate_box,['right'])
        self.accept('arrow_up',self.generate_box,['up'])
        
        square1 = create_rec(0,0,200,200)
        gnode = GeomNode('square')
        plane1 = self.render.attachNewNode(gnode)

        # 碰撞
        self.cTrav = CollisionTraverser()
        self.collHandEvent = CollisionHandlerEvent()
        self.collHandEvent.addInPattern('into-%in')

        self.combo = 0
        self.music = None
        
        self.clap_sound = self.loader.loadSfx("clap.wav")
        self.clap_sound.setVolume(0.5)
        
        self.combo_10 = self.loader.loadSfx('10combo.wav')
        self.combo_10.setVolume(0.5)
        
        self.combo_50 = self.loader.loadSfx('50combo.wav')
        self.combo_50.setVolume(0.5)
        
        self.combo_100 = self.loader.loadSfx('100combo.wav')
        self.combo_100.setVolume(0.5)
        
        
        self.floor = self.loader.loadModel('floor/floor.bam')
        self.floor.reparentTo(self.render)
        self.floor.setScale(10,0.4,1)
        self.floor.setPos(1.3,28,1.2)
        self.floor.setHpr(90,0,0)
        self.floor.setColor(0.5,0.5,1,1)
        
        
        self.floor = self.loader.loadModel('floor/floor.bam')
        self.floor.reparentTo(self.render)
        self.floor.setScale(10,1.8,1)
        self.floor.setPos(5.8,28,-1.5)
        self.floor.setHpr(90,0,0)
        self.floor.setColor(1,0,0,0)
        
        self.stick = self.loader.loadModel('output.bam')
        self.stick.reparentTo(self.render)
        self.stick.setScale(0.5,0.5,0.75)
        self.stick.setPos(0,20,-4)
        # self.stick.setColor(1,0,1,1)
        
        stick_tex = self.loader.loadTexture('stick/initialShadingGroup_baseColor.png')
        ts = TextureStage('ts')
        ts.setMode(TextureStage.MModulate)
        self.stick.setTexture(ts,stick_tex)
        
        
    
        
        stick_collision_node = CollisionNode("stick_collision")
        stick_collision_node.addSolid(CollisionBox((0,0,8.5),4,1,1))
        collider_stick = self.stick.attachNewNode(stick_collision_node)
        collider_stick.show()
        self.cTrav.addCollider(collider_stick, self.collHandEvent)
        self.accept('into-' + 'stick_collision', self.collideStick)
       
        
        
   
        finish_line = self.loader.loadModel('finsih.bam')
        finish_line.reparentTo(self.render)
        finish_line.setScale(1,0.5,0.2)
        finish_line.setPos(0,20,-1)
        finish_line.setColor(1, 1, 1, 0.5)
        
        finish_line2 = self.loader.loadModel('finsih.bam')
        finish_line2.reparentTo(self.render)
        finish_line2.setScale(1,0.5,0.2)
        finish_line2.setPos(0,50,-1)
        finish_line2.setColor(1, 1, 1, 0.5)
        
        finish_line3 = self.loader.loadModel('finsih.bam')
        finish_line3.reparentTo(self.render)
        finish_line3.setScale(1,0.5,0.2)
        finish_line3.setPos(0,80,-1)
        finish_line3.setColor(1, 1, 1, 0.5)

        finish_line_collision_node2 = CollisionNode("finish_line2_collision")
        finish_line_collision_node2.addSolid(CollisionBox((0,0,0),7,1,7))
        collider_finish_line2 = finish_line2.attachNewNode(finish_line_collision_node2)
        # collider_finish_line.show()
        self.cTrav.addCollider(collider_finish_line2, self.collHandEvent)
        self.accept('into-' + 'finish_line2_collision', self.collide2)
        
        finish_line_collision_node3 = CollisionNode("finish_line3_collision")
        finish_line_collision_node3.addSolid(CollisionBox((0,0,0),7,1,7))
        collider_finish_line3 = finish_line3.attachNewNode(finish_line_collision_node3)
        # collider_finish_line.show()
        self.cTrav.addCollider(collider_finish_line3, self.collHandEvent)
        self.accept('into-' + 'finish_line3_collision', self.collide3)
    

        finish_line_up = self.loader.loadModel('finsih.bam')
        finish_line_up.reparentTo(self.render)
        finish_line_up.setScale(0.25,0.5,0.2)
        finish_line_up.setPos(0,50,2)
        finish_line_up.setColor(1, 1, 0, 0.5)
        
        
        finish_line2_up = self.loader.loadModel('finsih.bam')
        finish_line2_up.reparentTo(self.render)
        finish_line2_up.setScale(0.25,0.5,0.2)
        finish_line2_up.setPos(0,80,2)
        finish_line2_up.setColor(1, 1, 0, 0.5)
        
        
        finish_line3_up = self.loader.loadModel('finsih.bam')
        finish_line3_up.reparentTo(self.render)
        finish_line3_up.setScale(0.25,0.5,0.2)
        finish_line3_up.setPos(0,20,2)
        finish_line3_up.setColor(1, 1, 0, 0.5)
        
        
        finish_line_up_collision_node = CollisionNode("finish_line_up_collision")
        finish_line_up_collision_node.addSolid(CollisionBox((0,0,0),7,1,4))
        collider_finish_line_up = finish_line_up.attachNewNode(finish_line_up_collision_node)
        # collider_finish_line_up.show()
        self.cTrav.addCollider(collider_finish_line_up, self.collHandEvent)
        self.accept('into-' + 'finish_line_up_collision', self.collide2)
        
        finish_line2_up_collision_node = CollisionNode("finish_line2_up_collision")
        finish_line2_up_collision_node.addSolid(CollisionBox((0,0,0),7,1,4))
        collider_finish_line2_up = finish_line2_up.attachNewNode(finish_line2_up_collision_node)
        # collider_finish_line_up.show()
        self.cTrav.addCollider(collider_finish_line2_up, self.collHandEvent)
        self.accept('into-' + 'finish_line2_up_collision', self.collide3)


        
        self.start_text_node = TextNode('start_node')
        self.start_text_node.setText('Press Spacebar to Start')
        onscreen_start_test = self.aspect2d.attachNewNode(self.start_text_node)
        onscreen_start_test.setScale(0.2)
        onscreen_start_test.setPos(-1,1,0.7)
        
        self.combo = 0
        self.best_combo = 0
        self.text_node = TextNode('text_node')
        self.text_node.setText('')
        onscreen_text = self.aspect2d.attachNewNode(self.text_node)
        onscreen_text.setScale(0.1)
        onscreen_text.setPos(-1.2, 0, 0.2)
        
        self.text_best_node = TextNode('text_best_node')
        self.text_best_node.setText('')
        onscreen_best_text = self.aspect2d.attachNewNode(self.text_best_node)
        onscreen_best_text.setScale(0.1)
        onscreen_best_text.setPos(-1.25, 0, 0.8)
        
        self.accept_once('space',self.gameStart,[onscreen_start_test])

        self.task_mgr.add(self.shared_rotate, 'shared_rotate')

        
        self.accept('d',self.rotate_stick)
        self.accept('d-repeat',self.rotate_stick)
        self.accept('a',self.rotate_stick2)
        self.accept('a-repeat',self.rotate_stick2)
        self.accept('w',self.rotate_stick3)
        self.accept('w-repeat',self.rotate_stick3)
        self.accept('s',self.rotate_stick4)
        self.accept('s-repeat',self.rotate_stick4)

        self.cam_to_panda = np.array([
            [-1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]
        ]).T

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


    def get_panda_pose(self, pose):
        return self.cam_to_panda @ pose


    def try_flip(self, top_vec):
        if top_vec[2] < 0:
            return -top_vec
        return top_vec


    def top_vec_to_R(self, top_vec, coor='panda'):
        basis = self.get_basis(coor)
        return rotation_matrix_from_vectors(basis, top_vec)
    

    def rotation_to_quat(self, R):
        return scipy.spatial.transform.Rotation.from_matrix(R).as_quat()

    def set_pose(self, target, R, t):
        target.setQuat(LQuaternionf(*self.rotation_to_quat(R)))
        target.setPos(*t)

    
    def shared_rotate(self, task):
        pose = self.shared_state['pose']
        if pose is None:
            return Task.cont

        pose = self.get_panda_pose(pose)

        R, t = self.pose_to_Rt(pose)
        top_vec = self.get_top_vec(R, 'panda')
        top_vec = self.try_flip(top_vec)
        R = self.top_vec_to_R(top_vec, 'panda')

        t = self.stick.getPos()

        self.set_pose(self.stick, R, t)

        return Task.cont

    
    def gameStart(self,node):
        # 创建OnscreenText，将文本添加到屏幕上
        self.text_node.setText('0 Combo !')
        self.text_best_node.setText('best: 0 combo')
        self.disableMouse()
        self.camera.setPos(0, -10, 12) 
        self.camera.setHpr(0, -10, 0)      
        node.removeNode()
        # self.play_music()
        objects_data = self.read_objects_data("easyboxes.txt")
        # print(objects_data)
        ref_time = globalClock.getFrameTime()
        self.task_mgr.add(self.music_task,"music_task",extraArgs=[ref_time])
        self.taskMgr.add(self.spawn_objects_task, "spawn_objects_task", extraArgs=[objects_data,ref_time])
        self.camera_pos = 1
        self.accept('c',self.change_camera_angle)
        self.taskMgr.add(self.combo_task, "combo_task")
        
    def music_task(self,ref_time):
        current_time = globalClock.getFrameTime()    
        if current_time - ref_time >= 2:
            self.play_music()
            return Task.done        
        return Task.cont
    
    def combo_task(self,task):
        if self.combo == 10:
            self.combo_10.play()
        if self.combo == 50:
            self.combo_50.play()
        if self.combo == 100:
            self.combo_100.play()
        return Task.cont
    
    def change_camera_angle(self):
        if self.camera_pos == 1:
            self.camera.setPos(0, -10, 0) 
            self.camera.setHpr(0, 0, 0)
            self.camera_pos = 2      
        else:
            self.camera.setPos(0, -10, 12) 
            self.camera.setHpr(0, -10, 0)
            self.camera_pos = 1  
             
    def rotate_stick(self):
        self.stick.setHpr(0,0,90)
    def rotate_stick2(self):
        self.stick.setHpr(0,0,-90)
    def rotate_stick3(self):
        self.stick.setHpr(0,0,0)
    def rotate_stick4(self):
        self.stick.setHpr(self.stick.getH(),self.stick.getP() - 10,self.stick.getR())
    
    
    def change_text(self):
        self.text_node.setText(f"{self.combo} Combo !")
        if self.combo > self.best_combo:
            self.best_combo = self.combo
            self.text_best_node.setText(f"best: {self.best_combo} combo")
    
    def collideStick(self,collEntry):
        self.clap_sound.play()
        collParent = collEntry.getFromNodePath().getParent()
        collParent.setColor(1,0,0,0)
        self.combo += 1
        self.change_text()
        

    
    def collide2(self,collEntry):
        collParent = collEntry.getFromNodePath().getParent()
        collParent.setColor(0,1,0,1)
    
    def collide3(self,collEntry):
        collParent = collEntry.getFromNodePath().getParent()
        collParent.setColor(0,1,1,1)
    
    
    def generate_box(self,arg1):
        box = self.loader.loadModel("models/box")
        box.reparentTo(self.render)
        box.setScale(1.5)
        if arg1 == 'left':
            box.setPos(-5,150,-1)
        elif arg1 == 'right':
            box.setPos(3.5,150,-1)
        else:
            box.setPos(-0.6,150,2)
        box_collision_node = CollisionNode("box_collision")
        box_collision_node.addSolid(CollisionBox(Point3(0.5, 0.5, 0.5), 0.65, 0.65, 0.65))
        collider_box = box.attachNewNode(box_collision_node)
        # collider_box.show()
        self.cTrav.addCollider(collider_box, self.collHandEvent)
        # self.accept('into-' + 'box_collision', self.collide)
        
        
        self.taskMgr.add(self.move_box_task, "move_box_task",extraArgs=[box])
        
    def move_box_task(self,box):
        # 移動箱子的任務函數
        goal = 5 
        speed = 40      # 移動速度
        rotation_speed = 50
        
        if box.hasColor():
            if box.getColor() == (1,0,0,0):
                box.removeNode()
                return Task.done


        # 檢查箱子是否已經移動到目標位置
        if box.getY() > goal:
            # 移動箱子
            box.setPos(box.getX(), box.getY() - speed* globalClock.getDt(), box.getZ())
            angle_degrees = rotation_speed * globalClock.getDt()
            # box.setHpr(box.getH() + angle_degrees,0, 0)
        else:
            # 如果箱子已經移動到目標位置，停止任務
            box.removeNode()
            self.combo = 0
            self.change_text()
            return Task.done

        return Task.cont
    
    def play_music(self):
        self.music = self.loader.loadMusic('bad_apple.mp3')
        self.music.play()
        self.music.setVolume(0.2)
    
    def onCollision(self, entry):
        print("Collision occurred!")
    
    def read_objects_data(self, file_path):
        objects_data = []
        try:
            with open(file_path, "r") as file:
                for line in file:
                    data = line.strip().split(',')
                    time = float(data[0])
                    object_type = data[1]
                    objects_data.append((time, object_type))
            return objects_data
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None
    
    def spawn_objects_task(self,objects_data,ref_time):
        current_time = globalClock.getFrameTime()
        for time, object_type in objects_data:
            if current_time - ref_time >= time - 0.5:
                self.generate_box(object_type)
                objects_data.remove((time, object_type))
        
        return Task.cont

if __name__ == '__main__':
    app = MyApp()
    app.run()