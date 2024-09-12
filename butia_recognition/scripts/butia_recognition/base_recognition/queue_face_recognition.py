import numpy as np
from PIL import Image
import face_recognition
from queue import Queue
import pyflann
from sklearn.neighbors import KNeighborsClassifier


class QueueFaceRecogNoDuplicate:
    def __init__(self, threshold=0.85, otimized='None', encodes=None, n_neighbors=5, algorithm='auto'):
        self.threshold = threshold
        self.method = otimized
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        
        self._loadSavedEncodings(encodes)
        
        if self.method == 'knn':
            self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, algorithm=self.algorithm)
            self._trainKNN()
        elif self.method == 'flann':
            self.flann = pyflann.FLANN()
        
    def _loadSavedEncodings(self, encondings_location) -> dict:
        self.saved_faces_encodes = encondings_location
        return 
    
    def _trainKNN(self):
        self.encodings, self.labels = [], []
        for label, encodings_list in self.saved_faces_encodes.items():
            self.labels.extend([label] * len(encodings_list))
            self.encodings.extend(encodings_list)
        self.knn.fit(self.encodings, self.labels)
        
        return
        
    def _processImage(self, image_frame: np.ndarray) -> dict:
        imagem = Image.fromarray(image_frame)
        faces_location = face_recognition.face_locations(np.array(imagem), model = 'yolov8')
        faces_encodings = face_recognition.face_encodings(np.array(imagem), faces_location)
        self.current_faces_encodes = {}
        
        for i in range(0 ,len(faces_encodings)):
            top, right, bottom, left = faces_location[i]
            self.current_faces_encodes[f'Rosto{i+1}'] = ( faces_encodings[i], [top, right, bottom, left] )

        return 
    
    def _calculateSimilarity(self, encoding1, encoding2) -> float:
        return np.dot(encoding1, encoding2) / (np.linalg.norm(encoding1) * np.linalg.norm(encoding2))
    
    def _sortFunc(self, item) -> float:
        return -item[0]
    
    def _queueSort(self, item: dict) -> dict:
        queue = {}
        for face, fila in item.items():
            queue[face] = Queue()
            for obj in fila:
                queue[face].put(obj)
        return queue
    
    def _sortCompare(self) -> dict:
        filas = {}
        for actual_face, encoding in self.current_faces_encodes.items():
            if actual_face not in filas:
                filas[actual_face] = []
                
            if self.method == 'knn':
                distances, indices = self.knn.kneighbors(encoding[0].reshape(1, -1), n_neighbors=self.n_neighbors)
                for distance, index in zip(distances[0], indices[0]):
                    similarity = 1 - distance
                    if similarity >= self.threshold:
                        sim_encode = [similarity, self.labels[index], encoding]
                        filas[actual_face].append(sim_encode)
            else:
                for label, saved_encodings in self.saved_faces_encodes.items():
                    for saved_encoding in saved_encodings:
                        if len(saved_encoding) != 0:
                            if self.method == 'None':
                                similar = self._calculateSimilarity(saved_encoding, encoding[0])
                                if similar > self.threshold:
                                    sim_encode = [similar, label, encoding]
                                    filas[actual_face].append(sim_encode)
                                else:
                                    continue
                            elif self.method == 'flann':
                                similar = self.flann.nn(saved_encoding.reshape(1, -1), encoding[0].reshape(1, -1), 1)
                                similarity = 1 - similar[1]
                            
                                if similarity >= self.threshold:
                                    sim_encode = [similarity, label, encoding]
                                    filas[actual_face].append(sim_encode)
                                else:
                                    continue
                        else:
                            pass

        sorted_filas = {key: sorted(value, key=self._sortFunc) for key, value in filas.items()}

        return sorted_filas
    
    def _resolveConflicts(self, sorted_filas):
        # Dicionário para armazenar temporariamente as faces com suas respectivas labels
        queued_filas = self._queueSort(sorted_filas)
        final_list = {}
        
        if len(queued_filas) == 0:
            return final_list
        
        elif len(queued_filas) == 1:
            for queued, saved in zip(queued_filas.items(), self.current_faces_encodes.items()):
                if queued[1].empty():
                    queued[1].put([[0.0], "Unknown", (saved[1][0], saved[1][1])])
                final_list[queued[0]] = queued[1].queue[0]
            return final_list
        
        else:
        
            labels_temporarias = {}
            
            # Percorre cada fila
            for face, fila in queued_filas.items():
                if not fila:
                    continue
                # Inicializa a face com a primeira entrada da fila
                if sorted_filas[face] == []:
                    labels_temporarias[face] = []
                else:
                    labels_temporarias[face] = sorted_filas[face][0]
            done = False
            c = 0
            while done == False:
                #print('')
                for q_face, q_fila in queued_filas.items():
                    #print('8')
                    for temp_face, temp_fila in labels_temporarias.items():
                        #print('9')
                        # Check if the face is not the same
                        if c > 10:
                            done = True
                        if not q_fila.empty() and not temp_fila == []:
                            #print('Fila not empty')
                            # Check if the queue is not empty and the temporary list is not empty
                            if q_face != temp_face:
                                #print('Different faces')
                                c+=1
                                # Check label repitition
                                if q_fila.queue[0][1] == temp_fila[1]:
                                    #print('Labels repeated')
                                    # Check similarity
                                    if q_fila.queue[0][0] < temp_fila[0]:
                                        #print('3')
                                        try:
                                            # Tries to remove the first element from the queue
                                            removed = q_fila.get_nowait()
                                        except:
                                            done = True
                                        
                                    else:
                                        # If the similarity is higher, the list is updated
                                        a = q_fila.queue[0]
                                        labels_temporarias[q_face] = a
                                else:
                                    pass
                            else:
                                continue
                        else:
                            done = True
                        
            for queued, saved in zip(queued_filas.items(), self.current_faces_encodes.items()):
                if queued[1].empty():
                    queued[1].put([[0.0], "Unknown", (saved[1][0], saved[1][1])])
                final_list[queued[0]] = queued[1].queue[0]
        # Retorna o dicionário de faces com as labels resolvidas
            return final_list
        
    def runFaceRecognition(self, image_path: str) -> dict:
        self._processImage(image_path)
        sorted_filas = self._sortCompare()
        resolved_labels = self._resolveConflicts(sorted_filas)
        return resolved_labels