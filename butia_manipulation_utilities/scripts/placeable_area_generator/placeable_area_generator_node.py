#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
from typing import List
from matplotlib import pyplot as plt
import numpy as np
import rospy
import cv2
import open3d as o3d
from butia_vision_msgs.srv import PlaceableArea, PlaceableAreaResponse, PlaceableAreaRequest
from scipy.spatial import ConvexHull
from visualization_msgs.msg import MarkerArray
from cv_bridge import CvBridge
from butia_vision_bridge import VisionBridge
import tf2_ros
from geometry_msgs.msg import Pose

class PlaceableAreaNode:
    def __init__(self, init_node=False):
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        self.target_label = 'Utelsils'
        self.shelf_id = 1
        self.radius = 0.1
        self.resolution = 1000
        self.step_size = 50
        self.justObjectImage = None
        self.tf2_buffer = tf2_ros.Buffer(rospy.Duration(1200.0))
        self.listener = tf2_ros.TransformListener(self.tf2_buffer)
        self.transform = self.tf2_buffer.lookup_transform('map', 'camera_color_optical_frame', rospy.Time(0), rospy.Duration(3.0))

        self.cv_bridge = CvBridge()
        self.initRosComm()

    def normalize(self, xy_hull):
        print("Normalizing")
        if self.x_min is None:
            self.x_min = np.min(xy_hull['x'])
            self.x_max = np.max(xy_hull['x'])
            self.y_min = np.min(xy_hull['y'])
            self.y_max = np.max(xy_hull['y'])

        xy_hull['x'] = ((xy_hull['x'] - self.x_min) / (self.x_max - self.x_min) * self.resolution + 0).astype(int)
        xy_hull['y'] = ((xy_hull['y'] - self.y_min) / (self.y_max - self.y_min)  * self.resolution + 0).astype(int)

        return xy_hull

    def denormalize(self, xy_hull):
        print("Denormalizing")
        xy_hull['x'] = ((xy_hull['x'] - 0) / self.resolution * (self.x_max - self.x_min) + self.x_min)
        xy_hull['y'] = ((xy_hull['y'] - 0) / self.resolution * (self.y_max - self.y_min) + self.y_min)
        return xy_hull

    
    def initRosComm(self):
        self.grasp_generator_srv = rospy.Service('butia_vision_msgs/placeable_area', PlaceableArea, self.callback)

    def callback(self, data: PlaceableAreaRequest):
        # Read cloud with o3d.io.read_point_cloud
        # point_cloud: PointCloud2 = data.point_cloud 
        markers: MarkerArray = data.markers

        #   plane = [
        #     [2.001011 -1.393388 -0.177392], # Esquerda trás
        #     [1.832372 -1.391613 -0.106312], # Esquerda frente
        #     [1.872976 -0.840142 -0.124781], # Direita frente
        #     [2.033025 -0.867732 -0.196236]  # Direita trás
        # ]

        plane = {
            'x': [2.001011, 1.832372, 1.872976, 2.033025], 'y': [-1.393388, -1.391613, -0.840142, -0.867732],
            }
        
        # TODO: check if the markers are aligned with the point cloud
        markers, others_objs = self.transform_markers_in_pcds(markers)
        
        t_cloud: o3d.geometry.PointCloud = VisionBridge.pointCloud2XYZRGBtoOpen3D(point_cloud)

        # Transform the cloud to the map frame
        cloud = self.transform_pcd(copy.deepcopy(t_cloud), self.transform)    

        # Segment the plane
        plane_model, inliers = self.filter_normals(cloud) # a*x + b*y + c*z + d = 0

        markers = [self.transform_pcd(copy.deepcopy(marker), self.transform) for marker in markers]    
        
        for label, pcds in others_objs.items():
            others_objs[label] = [self.transform_pcd(copy.deepcopy(pcd), self.transform) for pcd in pcds]

        
        markers = self.remove_above_plane_pcd(plane_model, markers)

        for label, pcds in others_objs.items():
            others_objs[label] = self.remove_above_plane_pcd(plane_model, pcds)

        union_markers = o3d.geometry.PointCloud()
        for marker in markers:
            union_markers += marker  

        x_center = np.mean(np.asarray(union_markers.points)[:, 0])
        y_center = np.mean(np.asarray(union_markers.points)[:, 1])
        z_center = np.mean(np.asarray(union_markers.points)[:, 2])
        
        union_markers_others = o3d.geometry.PointCloud()
        for label, pcds in others_objs.items():
            for pcd in pcds:
                union_markers_others += pcd

        # get just the object
        hull_obj = self.get_hull(union_markers)

        hull_all = copy.deepcopy(hull_obj)

        for label, pcds in others_objs.items():
            for pcd in pcds:
                hull_all += self.get_hull(pcd)

        try:
            obj_2d = np.asarray(hull_obj.points)
        except:
            obj_2d = np.asarray(hull_obj)
                
        xy_hull = self.get_xy_by_hull(obj_2d)
        xy_others_hull = []
        for label, pcds in others_objs.items():
            for pcd in pcds:
                try:
                    obj_2d = np.asarray(pcd.points)
                except:
                    obj_2d = np.asarray(pcd)
                xy_others_hull.append(self.get_xy_by_hull(obj_2d))
                
        
        
        xy_hull_all = dict()
        xy_hull_all['x'] = []
        xy_hull_all['y'] = []
        for xy in xy_others_hull:
            for x in xy['x']:
                xy_hull_all['x'].append(x)
            for y in xy['y']:
                xy_hull_all['y'].append(y)
        
        for x in xy_hull['x']:
            xy_hull_all['x'].append(x)
        
        for y in xy_hull['y']:
            xy_hull_all['y'].append(y)

        
        self.normalize(plane)

        self.normalize(xy_hull_all)

        # Normalize the xy_hull coordinates
        xy_hull_all = copy.deepcopy(xy_others_hull)
        xy_hull_all.append(xy_hull)

        xy_hull_all_normalized = []
        for xy in xy_hull_all:
            xy_hull_all_normalized.append(self.normalize(xy))    
        
        
        # xy_hull_all = self.normalize(xy_hull_all)

        # TODO: Get image by an array of hulls for all object in one image
        image = self.from_xy_to_img(xy_hull_all_normalized)
        print('aqui foi')


        # TODO: Center for every hull of the same label
        print('oppp')
        hull_center = ((np.mean(xy_hull['x'])).astype(int), (np.mean(xy_hull['y']).astype(int)))

        placeable_point_xy, visited = self.bfs(image, hull_center)
        print('opp1111p')

        if placeable_point_xy is None:
            print('No placeable point found')
            response = PlaceableAreaResponse()
            response.score = 0
            response.id = 0
            return response
        
        self.debug_image(image, placeable_point_xy, visited)
        
        # Reverter a normalização
        placeable_point_xy = self.denormalize({'x': placeable_point_xy[0], 'y': placeable_point_xy[1]})
        placeable_point_np = np.array([placeable_point_xy['x'], placeable_point_xy['y']])

        # Adicionar uma terceira coordenada ao vertices
        vertices_list = [np.append(placeable_point_np, z_center)]  # Adicionar 0 como a coordenada z

        central_placeable = o3d.geometry.PointCloud()
        central_placeable.points = o3d.utility.Vector3dVector(vertices_list)
        central_coordinates = np.asarray(central_placeable.points)[0]
        print("Central coordinates", central_coordinates)
        print("Original coordinates", x_center, y_center, z_center)
        # Transformar o ponto central em uma esfera
        sphere, projected_sphere = self.point_to_sphere(plane_model, central_coordinates)
        

        # Visualizar a esfera e outros objetos
        self.visualize_3d([union_markers, projected_sphere, sphere, hull_all, cloud])

        response = PlaceableAreaResponse()

        pose = Pose()
        pose.position.x = central_coordinates[0]
        pose.position.y = central_coordinates[1]
        pose.position.z = central_coordinates[2]

        response.response = pose

        return response

    def remove_above_plane_pcd(self, plane_model, markers):
            # Check if each marker is above the plane
            remove_indexes = []
            for i in range(len(markers)):
                markers[i] = self.crop_pcd_above_plane(plane_model, markers[i])
                if len(markers[i].points) == 0:
                    remove_indexes.append(i)

            tempMarkers = []

            for i in range(len(markers)):
                if i not in remove_indexes:
                    tempMarkers.append(markers[i])

            return tempMarkers
    
    def point_to_sphere(self, plane_model, central_coordinates):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.radius * 0.5)
        sphere.translate(central_coordinates)  # Transladar a esfera para o ponto central

        # Converter a esfera (TriangleMesh) em uma PointCloud
        sphere_pcd = sphere.sample_points_uniformly(number_of_points=1000)

        projected_sphere = self.get_projected_cloud(plane_model, sphere_pcd)
        return sphere, projected_sphere
    
    def transform_markers_in_pcds(self, markers: MarkerArray):
        others_pcds = dict()
        target_pcds = []
        for marker in markers.markers:
            if marker.text == '':
                continue

            # get the conf from text text: "Pantry items/mustard (0.81)"
            conf = float(marker.text.split('(')[1].split(')')[0])
            if conf < 0.5:
                continue

            # Crie os vértices do paralelogramo
            vertices = np.array([
                [0, 0, 0],
                [marker.scale.x, 0, 0],
                [marker.scale.x, marker.scale.y, 0],
                [0, marker.scale.y, 0],
                [0, 0, marker.scale.z],
                [marker.scale.x, 0, marker.scale.z],
                [marker.scale.x, marker.scale.y, marker.scale.z],
                [0, marker.scale.y, marker.scale.z]
            ])

            # Crie uma malha a partir dos vértices
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector([
                [0, 1, 2], [2, 3, 0],  # Bottom face
                [4, 5, 6], [6, 7, 4],  # Top face
                [0, 1, 5], [5, 4, 0],  # Front face
                [2, 3, 7], [7, 6, 2],  # Back face
                [0, 4, 7], [7, 3, 0],  # Left face
                [1, 5, 6], [6, 2, 1]   # Right face
            ])

            # Amostre pontos uniformemente na malha
            pcd = mesh.sample_points_uniformly(number_of_points=1000)

            # Aplique a transformação do marcador à nuvem de pontos
            transform = np.eye(4)
            transform[:3, 3] = [marker.pose.position.x, marker.pose.position.y, marker.pose.position.z]
            pcd.transform(transform)

            label = marker.text.split('/')[0]

            
            if label == self.targe_label:
                target_pcds.append(pcd)
            else:
                if label not in others_pcds:
                    others_pcds[label] = []
                print(label, others_pcds)
                others_pcds[label].append(pcd)

        return target_pcds, others_pcds

    def transform_pcd(self, pcd, transform):
        t = transform.transform
        translation = t.translation
        rotation = t.rotation
        pcd.translate([translation.x, translation.y, translation.z])
        R = pcd.get_rotation_matrix_from_quaternion([rotation.x, rotation.y, rotation.z, rotation.w])
        pcd.rotate(R, center=(0, 0, 0))    
        return pcd
    
    def filter_normals(self, pcd):
        downpcd = pcd.voxel_down_sample(voxel_size=0.049)

        downpcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
        filtered_points = []
        filtered_normals = []

        for idx, normal in enumerate(downpcd.normals):
            if normal[2] > 0.9 or normal[2] < -1:
                filtered_points.append(downpcd.points[idx])
                filtered_normals.append(downpcd.normals[idx])
        pcd_filtered = o3d.geometry.PointCloud()
        pcd_filtered.points = o3d.utility.Vector3dVector(filtered_points)
        pcd_filtered.normals = o3d.utility.Vector3dVector(filtered_normals)

        # pcd_filtered = pcd_filtered.translate((0, 0, 0.1))

        pcd_filtered, _ = pcd_filtered.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=0.5)
        
        plane_model, inliers = pcd_filtered.segment_plane(distance_threshold=0.01,
                        ransac_n=3,
                        num_iterations=1000)
        
        points = np.asarray(pcd.points)  
        indexes_to_paint = []

        # Color points above the plane in red
        for i in range(len(points)):
            if self.is_point_above_plane(points[i], plane_model):
                try:
                    indexes_to_paint.append(i)
                except Exception as e:
                    print(pcd.colors)
                    print(f"An error occurred: {e}")
                    raise e
        
        temp = pcd.select_by_index(indexes_to_paint)
        temp.paint_uniform_color([1, 0, 0])

        # o3d.visualization.draw_geometries([pcd, temp])

        return plane_model, inliers
    
    def is_point_above_plane(self, point, plane_model):
        [a, b, c, d] = plane_model
        [x, y, z] = point

        return a*x + b*y + c*z + d < 0

    def crop_pcd_above_plane(self, plane_model, pcd):
        [a, b, c, d] = plane_model
        points = np.asarray(pcd.points)
        indexes_to_crop = []

        for i in range(len(points)):
            if self.is_point_above_plane(points[i], plane_model):
                indexes_to_crop.append(i)

        new_pcd = pcd.select_by_index(indexes_to_crop)

        return new_pcd

    def get_projected_cloud(self, plane_model: List[float], cloud: o3d.geometry.PointCloud):
        [a, b, c, d] = plane_model
        projected_points = []
        for point in np.array(cloud.points):
            t = -(a*point[0] + b*point[1] + c*point[2] + d) / (a*a + b*b + c*c)
            projected_point = point + np.array([a, b, c]) * t
            projected_points.append(projected_point)
        projected_cloud = o3d.geometry.PointCloud()
        projected_cloud.points = o3d.utility.Vector3dVector(projected_points)
        return projected_cloud
    
    def clustering(self, cloud: o3d.geometry.PointCloud):
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(
                cloud.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
        colors = cloud.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

        return cloud

    def extract_obj(self, cloud):
        obj_color = np.array([0.83921569, 0.15294118, 0.15686275])

        tolerance = 0.1
        colors = np.asarray(cloud.colors)
        indices = np.where(np.linalg.norm(colors - obj_color, axis=1) < tolerance)[0]

        obj_cloud = cloud.select_by_index(indices)

        # o3d.visualization.draw_geometries([obj_cloud])
        return obj_cloud

    def get_hull(self, cloud):
        hull, _ = cloud.compute_convex_hull()
        hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
        hull_ls.paint_uniform_color((1, 0, 0))
        return hull_ls
    
    def to2d(self, plane_model: List[float], cloud: o3d.geometry.PointCloud):
        # Calculate the plane orientation
        # plane_normal = np.array([plane_model[0], plane_model[1], plane_model[2]])
        # plane_orientation = np.arccos(plane_normal.dot([0, 0, 1]))

        # # Calculate the rotation matrix to align the plane with the XY plane
        # rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz((plane_orientation, 0, 0))

        # # Project the points of the point cloud onto the XY plane
        try:
            points = np.asarray(cloud.points)
        except:
            points = np.asarray(cloud)
        # points = points.dot(rotation_matrix.T)  # Transpose the rotation matrix to undo the rotation

        return points

    def to3d(self, plane_model, vertices):
        # Calculate the plane orientation
        # plane_normal = np.array([plane_model[0], plane_model[1], plane_model[2]])
        # plane_orientation = np.arccos(plane_normal.dot([0, 0, 1]))

        # # Calculate the rotation matrix to align the XY plane with the original plane
        # rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, -plane_orientation))

        # # Transform the vertices back to the 3D world
        # print("Vertices", vertices, vertices.shape)
        # vertices += plane_model[:3] * plane_model[3]  # Apply the translation
        # vertices = vertices.dot(rotation_matrix.T)  # Apply the rotation

        vertices_list = [vertices]

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(vertices_list)

        return cloud

    def get_hull2d(self, points: List[List[float]]):
        # Calculate the 2D convex hull
        hull = ConvexHull(points[:, :2])

        # Return the indices of the points that form the convex hull
        return hull.vertices
    
    def get_xy_by_hull(self, points):

        hull_indices = self.get_hull2d(points)

        hull_points = points[hull_indices]
        # hull_points = np.concatenate([hull_points, hull_points[0:1]])

        # Plotar os pontos e a casca convexa
        plt.figure()
        plt.plot(points[:, 0], points[:, 1], 'o')
        plt.plot(hull_points[:, 0], hull_points[:, 1], 'r--', lw=2)
        plt.savefig('hull.png')
        

        edge = {}
        edge['x'] = hull_points[:, 0]
        edge['y'] = hull_points[:, 1]
        return edge

    def from_xy_to_img(self, xy_hull):
        # Criar uma imagem em branco
        image = np.ones((1000, 1000), dtype=np.uint8) * 255
        for xy in xy_hull:
            # Normalizar os pontos do polígono para caber na imagem
            hull_points = np.array([xy['x'], xy['y']]).T
            hull_points = hull_points.astype(int)

            # Preencher os pontos dentro do polígono com preto
            for y in range(image.shape[0]):
                for x in range(image.shape[1]):
                    if self.point_in_hull((x, y), hull_points):
                        image[y, x] = 0
        # # Normalizar os pontos do polígono para caber na imagem
        # hull_points = np.array([xy_hull['x'], xy_hull['y']]).T
        # hull_points = hull_points.astype(int)

        # # Preencher os pontos dentro do polígono com preto
        # for y in range(image.shape[0]):
        #     for x in range(image.shape[1]):
        #         if self.point_in_hull((x, y), hull_points):
        #             image[y, x] = 0 

        plt.imshow(image, cmap='gray')
        plt.savefig('justObjects.png')

        self.justObjectImage = copy.deepcopy(image)

        return image
    
    def point_in_hull(self, point, hull):
        return cv2.pointPolygonTest(hull, point, False) >= 0
    
    def bfs(self, image, hull_center):
        # Initialize queue with the starting point (hull_center)
        queue = [(hull_center[1], hull_center[0])]
        queue = [hull_center]
        print('hull_center', hull_center, queue)

        # Initialize visited matrix with zeros (same size as the image)
        visited = np.zeros(image.shape, dtype=np.uint8)
        points_visited = []
        i = 0
        while queue:
            i += 1
            # Remove the first point from the queue
            point = queue.pop(0)
            if point[0] < 0 or point[1] < 0 or point[0] >= image.shape[0] or point[1] >= image.shape[1]:
                continue
            print('point' , i, point)

            # Check if the point has already been visited
            if visited[point] == 0:
                # Mark the point as visited
                visited[point] = 1

                # Check if the point is placeable
                # TODO: Put the radius as a parameter of the real object
                print('point', point)
                print(image[point])
                if self.is_placeable_point(point, image):
                    return point, points_visited

                points_visited.append(point)
                # Add neighbors to the queue (check image boundaries)
                if point[0] > 0: queue.append((point[0]-self.step_size, point[1]))
                # if point[0] < image.shape[0]-1: queue.append((point[0]+self.step_size, point[1])) // Comentado porque ignora passos para tras
                if point[1] > 0: queue.append((point[0], point[1]-self.step_size))
                if point[1] < image.shape[1]-1: queue.append((point[0], point[1]+self.step_size))

        # If breadth-first search doesn't find a placeable point, return None
        return None, visited

    def is_placeable_point(self, point, image):
        if image[point] == 0:
            return False
        # Calculate the radius size in pixels
        radius_pixels = int((self.radius + 0.04) * self.resolution * 1.5)

        # Check all points within a square of side 2*radius centered on the point
        for dy in range(-radius_pixels, radius_pixels+1):
            for dx in range(-radius_pixels, radius_pixels+1):
                # Check if the point is inside the image
                if 0 <= point[1]+dy < image.shape[1] and 0 <= point[0]+dx < image.shape[0]:
                    # If the point is black, the circle doesn't fit
                    if image[point[0]+dx, point[1]+dy] == 0:
                        return False

        # If no black point is found, the circle fits
        return True

    def debug_image(self, image, point, visited):
        # Calculate the radius size in pixels
        radius_pixels = int(self.radius * self.resolution * 1.5) 

        # Create a copy of the image to draw on
        debug_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)

        for point in visited:
            cv2.circle(debug_image, (point[0], point[1]), radius_pixels, (50, 50, 50), -1)

        # Fill the circle with red color
        # Invert the coordinates of the point
        cv2.circle(debug_image, (point[0], point[1]), radius_pixels, (0, 0, 255), -1)
        cv2.circle(debug_image, (point[0], point[1]), 5, (255, 0, 0), -1)

        print('oii')
        print(self.justObjectImage.shape[0])

        for y in range(self.justObjectImage.shape[0]):
            for x in range(self.justObjectImage.shape[1]):
                if self.justObjectImage[y, x] == 0:
                    debug_image[y, x] = 0

        plt.imshow(debug_image, cmap='gray')
        plt.savefig('allObjects+placeable place.png')

        return debug_image

    def visualize_3d(self, pcds):
        o3d.visualization.draw_geometries(pcds)

if __name__ == '__main__':
    rospy.init_node('placeable_area_node', anonymous = False)
    placeable_area = PlaceableAreaNode()
    rospy.spin()
    rospy.on_shutdown(placeable_area.shutdown)