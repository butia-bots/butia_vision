#!/usr/bin/env python

from butia_recognition import BaseRecognition, ifState

import rospy
import torch

from pathlib import Path

from boxmot.tracker_zoo import create_tracker
from ultralytics.yolo.engine.model import YOLO, TASK_MAP
from ultralytics.yolo.utils import IterableSimpleNamespace, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.files import increment_path
from utils.multi_yolo_backend import MultiYolo


class YoloTrackerRecognition(BaseRecognition):
    def __init__(self,state = True):
        super().__init__(state=state)
        self.readParameters()
        self.loadModel()
        self.initRosComm()
        rospy.logwarn("Finished starting")
    
    def serverStart(self, req):
        self.loadModel()
        return super().serverStart(req)
    
    def serverStop(self, req):
        self.unLoadModel()
        pass    

    def loadModel(self):
        self.model = YOLO(self.model_file)
        overrides = self.model.overrides.copy()
        self.model.predictor = TASK_MAP[self.model.task][3](overrides=overrides, _callbacks=self.model.callbacks)
        
        # extract task predictor
        self.predictor = self.model.predictor

        # combine default predictor args with custom, preferring custom
        combined_args = {**self.predictor.args.__dict__}
        # overwrite default args
        self.predictor.args = IterableSimpleNamespace(**combined_args)

        # setup source and model
        if not self.predictor.model:
            self.predictor.setup_model(model=self.model.model, verbose=False)
        # print(self.predictor.args.source)
        # self.predictor.setup_source(self.predictor.args.source)
        
        self.predictor.args.imgsz = check_imgsz(self.predictor.args.imgsz, stride=self.model.model.stride, min_dim=2)  # check image size
        self.predictor.save_dir = increment_path(Path(self.predictor.args.project) / self.predictor.args.name, exist_ok=self.predictor.args.exist_ok)
        
        # Check if save_dir/ label file exists
        if self.predictor.args.save or self.predictor.args.save_txt:
            (self.predictor.save_dir / 'labels' if self.predictor.args.save_txt else self.predictor.save_dir).mkdir(parents=True, exist_ok=True)
        # Warmup model
        if not self.predictor.done_warmup:
            self.predictor.model.warmup(imgsz=(1 if self.predictor.model.pt or self.predictor.model.triton else self.predictor.dataset.bs, 3, *self.predictor.imgsz))
            self.predictor.done_warmup = True
        self.predictor.seen, self.predictor.windows, self.predictor.batch, self.predictor.profilers = 0, [], None, (ops.Profile(), ops.Profile(), ops.Profile(), ops.Profile())
        self.predictor.add_callback('on_predict_start', self.on_predict_start)
        self.predictor.run_callbacks('on_predict_start')
        self.model = MultiYolo(
            model=self.model.predictor.model,
            device=self.predictor.device,
            args=self.predictor.args
        )
    
    def unLoadModel(self):
        del self.model
    
    @ifState
    def callback(self, *args):
        pass

    def readParameters(self):
        self.debug_topic = rospy.get_param("~publishers/debug/topic","/butia_vision/br/debug")
        self.debug_qs = rospy.get_param("~publishers/pose_recognition/topic",1)

        self.pose_recognition_topic = rospy.get_param("~publishers/pose_recognition/topic","/butia_vision/br/pose_detection")
        self.pose_recognition_topic = rospy.get_param("~publishers/pose_recognition/queue_size",1)

        self.threshold = rospy.get_param("~threshold", 0.5)

        self.model_file = rospy.get_param("~model_file","yolov8n-pose")

        self.config_path = rospy.get_param("~tracker_config_path", "")

        super().readParameters()
    
    def on_predict_start(self, predictor):
        predictor.trackers = []
        predictor.tracker_outputs = [None] * predictor.dataset.bs
        predictor.args.tracking_config = self.config_path
        for i  in range(predictor.dataset.bs):
            tracker = create_tracker(
                predictor.args.tracking_method,
                predictor.args.tracking_config,
                predictor.args.reid_model,
                predictor.device,
                predictor.args.half
            )
            predictor.trackers.append(tracker)


if __name__ == "__main__":
    rospy.init_node("yolo_tracjer_recognition_node", anonymous = True)
    yolo = YoloTrackerRecognition()
    rospy.spin()