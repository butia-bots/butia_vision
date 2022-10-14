import torchreid
import torch.nn.functional as F
import numpy as np

def to_tlbr(ret):
    ret[2:] += ret[:2]
    return ret

class PeopleReId:
    def __init__(self):
        self.frame = None
        self.is_tracking = False
        #TODO: baixar o modelo daqui: https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html, minha sugestao e nao comitar o modelo no repositorio, pois na minha experiencia pessoal, se for um modelo muito grande, pode bugar o git.
        self.feature_extractor = torchreid.utils.FeatureExtractor(model_name='osnet_x1_0', model_path='/home/butiabots/Workspace/osnet_x1_0_imagenet.pth')

    def setFrame(self, image_header, header, frame_id, frame):
        self.image_header = image_header
        self.header = header
        self.frame_id = frame_id
        self.frame = frame

    def setDetections(self, descriptions):
        self.descriptions = descriptions

    def startTrack(self):
        while self.frame is None:
            pass
        self.setPersonToMatch(self.frame, self.descriptions)
        self.is_tracking = True
    
    def stopTrack(self):
        self.gallery_features = None
        self.gallery_matrix = None
        self.is_tracking = False

    def reid(self):
        descriptions = self.descriptions
        dets = []
        if self.is_tracking:
            if len(descriptions) == 0:
                print("No detections")
                return []
            detections = np.array([(int(d.bbox.center.x-d.bbox.size_x/2), int(d.bbox.center.y-d.bbox.size_y/2), d.bbox.size_x, d.bbox.size_y) for d in descriptions], dtype=np.int32)
            match_id = self.reidPersons(self.frame, detections)
            print('id', match_id)
            dets.append(descriptions[match_id])
        return dets

    def setPersonToMatch(self, frame, descriptions):
        detections = np.array([(int(d.bbox.center.x-d.bbox.size_x/2), int(d.bbox.center.y-d.bbox.size_y/2), d.bbox.size_x, d.bbox.size_y) for d in descriptions], dtype=np.int32)
        max_bb = None
        max_area = 0.0
        for detection in detections:
            area = detection[2]*detection[3]
            if area > max_area:
                max_bb = detection
                max_area = area
        max_bb = to_tlbr(max_bb)
        person_image = frame[max_bb[1]:max_bb[3], max_bb[0]:max_bb[2]]
        self.gallery_features = self.feature_extractor([person_image,])
        #normalize features
        #self.gallery_features = F.normalize(gallery_features, p=2, dim=1)
        self.gallery_matrix = torchreid.metrics.compute_distance_matrix(self.gallery_features, self.gallery_features)
        self.is_tracking = True
        self.person_image = person_image

    def reidPersons(self, frame, detections):
        person_images = []
        for detection in detections:
            print('detection', detection)
            print('img_shape', frame.shape)
            detection = to_tlbr(detection)
            person_images.append(frame[detection[1]:detection[3], detection[0]:detection[2]])
        self.person_images = person_images
        query_features = self.feature_extractor(person_images)
        #normalize
        #query_features = F.normalize(query_features, p=2, dim=1)
        #compute distance
        #distance is a np array of shape (num_query_images, num_gallery_images)
        distance = torchreid.metrics.compute_distance_matrix(query_features, self.gallery_features).cpu()
        #TODO: setar isso aqui e o normalize por parametros
        reranking = False
        if reranking:
            query_matrix = torchreid.metrics.compute_distance_matrix(query_features, query_features)
            distance = torchreid.utils.re_ranking(distance, query_matrix.cpu(), self.gallery_matrix.cpu())
        #talvez aqui tenha que mudar para axis=1
        print(distance)
        closest_match_id = np.argmin(distance, axis=1)
        return closest_match_id[0]
